import argparse
import cv2
import time
import threading
import torch
from collections import deque
from robomaster import robot
from ultralytics import YOLO

# ---------- 全局配置参数 ----------
TRANSPORT_TURN_ANGLE = 45  # 机器人抓取物体后，为给摄像头提供更好视野而旋转的角度
CONFIRM_ARRIVAL_COUNT = 5  # 判定机器人到达目标点所需连续满足条件的帧数
PIXEL_ALIGN_TOLERANCE = 35  # 机器人对准视觉标签时的中心像素容差
g_ir_distance = None  # 存储红外传感器距离的回调变量
g_detected_markers = []  # 存储视觉标签检测结果的回调变量
g_marker_lock = threading.Lock()  # 保护 g_detected_markers 的线程锁
g_cam_width, g_cam_height = 0, 0  # 存储摄像头分辨率的全局变量


# ---------- 传感器与视觉回调函数 ----------
def infrared_sensor_update(sub_info):
    """(回调) 更新全局红外距离变量"""
    global g_ir_distance
    distance = sub_info[0]
    g_ir_distance = distance if distance and distance > 0 else None


class VisionMarker:
    """对视觉标签(Marker)信息的封装"""

    def __init__(self, x, y, w, h, info_text):
        self._norm_x = x
        self._norm_w = w
        self._text = info_text
        # 将归一化坐标转换为屏幕像素坐标
        self.corner1 = (int((x - w / 2) * g_cam_width), int((y - h / 2) * g_cam_height))
        self.corner2 = (int((x + w / 2) * g_cam_width), int((y + h / 2) * g_cam_height))

    @property
    def center_x(self): return self._norm_x

    @property
    def width(self): return self._norm_w

    @property
    def info(self): return self._text


def vision_marker_update(marker_data):
    """(回调) 更新全局视觉标签列表"""
    global g_detected_markers
    with g_marker_lock:
        g_detected_markers.clear()
        for data_item in marker_data:
            x, y, w, h, info = data_item
            g_detected_markers.append(VisionMarker(x, y, w, h, info))


# ---------- 主程序 ----------
def main(cam_w, cam_h, model_w, model_h, conf_thresh, video_res):
    global g_ir_distance, g_detected_markers, g_cam_width, g_cam_height
    g_cam_width, g_cam_height = cam_w, cam_h  # 初始化全局摄像头尺寸

    # ---------- 内部辅助函数 ----------
    def render_detections(image, detections, class_names, m_w, m_h, c_w, c_h):
        """在画面上绘制YOLO检测框"""
        display_img = image.copy()
        if not detections or not hasattr(detections, "boxes") or len(detections.boxes) == 0:
            return display_img
        for box in detections.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            scale_x, scale_y = c_w / m_w, c_h / m_h
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[class_id]} {confidence:.2f}"
            cv2.putText(display_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return display_img

    # ---------- 初始化模块 ----------
    # 1. 初始化YOLO模型
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = r"/home/nvidia/Desktop/rm_demo/best_wjh3.pt"
    print(f"正在从 {model_file} 加载YOLO模型...")
    yolo_model = YOLO(model_file)
    yolo_model.fuse()
    yolo_model.to(compute_device)
    yolo_class_names = yolo_model.names
    print(f"模型加载成功，使用的计算设备: {compute_device}")
    print("模型支持的类别:", yolo_class_names)

    # [ADDED] 新增代码块: 动态查找'bottle'的类别ID
    bottle_class_id = -1
    for class_id, class_name in yolo_class_names.items():
        if class_name.lower() == 'bottle':  # 使用 .lower() 忽略大小写，更稳健
            bottle_class_id = class_id
            break

    if bottle_class_id == -1:
        print("致命错误：在模型类别中未找到 'bottle'。请检查模型文件或类别名称。")
        return  # 退出程序
    else:
        print(f"成功找到 'bottle' 对应的类别ID为: {bottle_class_id}")
    # [ADDED] 新增代码块结束

    # 2. 初始化机器人连接
    master_robot = robot.Robot()
    print("机器人系统连接初始化...")
    master_robot.initialize(conn_type="ap")
    print("机器人连接建立成功。")
    chassis_ctrl = master_robot.chassis
    arm_ctrl = master_robot.robotic_arm
    gripper_ctrl = master_robot.gripper
    sensor_ctrl = master_robot.sensor
    vision_ctrl = master_robot.vision
    camera_ctrl = master_robot.camera

    # 3. 设置机器人初始状态
    print("移动机械臂至初始搜索姿态...")
    arm_ctrl.moveto(x=180, y=90).wait_for_completed()
    gripper_ctrl.open(power=50)
    print("已移动至初始搜索姿态。")

    # 4. 启动视频流
    time.sleep(2)
    print("正在开启摄像头视频流...")
    camera_ctrl.start_video_stream(display=False, resolution=video_res)
    print("视频流已成功启动。")
    frame_buffer = deque(maxlen=1)
    is_thread_active = True

    def capture_frame_loop():
        """后台线程，持续将最新帧放入缓冲区"""
        while is_thread_active:
            img = camera_ctrl.read_cv2_image(strategy="newest", timeout=1.0)
            if img is not None:
                frame_buffer.append(img)

    capture_thread = threading.Thread(target=capture_frame_loop, daemon=True)
    capture_thread.start()

    # 5. 初始化任务状态变量
    frame_process_counter = 0  # 记录已处理的帧数，用于延时启动某些模块
    latest_yolo_result = None  # 存储最新的YOLO推理结果
    is_ir_subscribed = False  # 标记红外传感器订阅状态
    is_marker_subscribed = False  # 标记视觉标签订阅状态
    bottle_is_secured = False  # 标记当前是否已抓取瓶子 (核心状态)
    placement_count = 0  # 已成功放置的瓶子计数
    last_marker_timestamp = 0  # 最后一次检测到有效标签的时间
    tracked_marker = None  # 当前正在追踪的视觉标签对象
    MARKER_LOST_DURATION = 0.75  # 视觉标签丢失被判为丢失的阈值(秒)
    MARKER_WIDTH_AT_TARGET = 0.25  # 到达目标时，标签的归一化宽度
    arrival_confirm_counter = 0  # 用于确认到达状态的帧计数器

    try:
        sensor_ctrl.set_distance_sensor(1)
    except Exception as e:
        print(f"提醒: 红外传感器设置失败，可能影响近距离判断: {e}")
        pass

    # ---------- 主循环 ----------
    try:
        while placement_count < 2:
            # 1. 获取最新帧
            current_frame = frame_buffer[-1] if len(frame_buffer) else None
            if current_frame is None:
                time.sleep(0.01)
                continue

            # 2. 预处理帧
            if current_frame.shape[1] != cam_w or current_frame.shape[0] != cam_h:
                current_frame = cv2.resize(current_frame, (cam_w, cam_h))

            display_frame = current_frame.copy()

            # ========================= 状态一: 搜索并抓取瓶子 =========================
            if not bottle_is_secured:
                model_input_frame = cv2.resize(current_frame, (model_w, model_h))
                results_stream = yolo_model(model_input_frame, conf=conf_thresh, device=compute_device, stream=True)
                for res in results_stream:
                    latest_yolo_result = res
                    display_frame = render_detections(current_frame, latest_yolo_result, yolo_class_names, model_w,
                                                      model_h, cam_w, cam_h)
                    frame_process_counter += 1
                    break

                if frame_process_counter > 20:  # 等待几帧稳定后再启动传感器
                    if not is_ir_subscribed:
                        sensor_ctrl.sub_distance(freq=5, callback=infrared_sensor_update)
                        is_ir_subscribed = True

                    if latest_yolo_result and hasattr(latest_yolo_result, "boxes") and len(
                            latest_yolo_result.boxes) > 0:
                        # [MODIFIED] 使用动态获取的 bottle_class_id 来筛选瓶子
                        best_bottle = max((b for b in latest_yolo_result.boxes if int(b.cls[0]) == bottle_class_id),
                                          key=lambda b: float(b.conf[0]), default=None)
                        if best_bottle:
                            x1, _, x2, _ = best_bottle.xyxy[0].cpu().numpy()
                            center_x_px = ((x1 + x2) / 2) * (cam_w / model_w)
                            offset_px = center_x_px - (cam_w / 2)

                            if abs(offset_px) > 30:  # 旋转对准
                                chassis_ctrl.drive_speed(x=0, y=0, z=(offset_px / cam_w) * 100)
                            else:  # 前进靠近
                                chassis_ctrl.drive_speed(x=0, y=0, z=0)
                                if g_ir_distance and g_ir_distance > 60:
                                    chassis_ctrl.drive_speed(x=0.07, y=0, z=0)
                                elif g_ir_distance:  # 距离足够近，执行抓取
                                    chassis_ctrl.drive_speed(x=0, y=0, z=0)
                                    if is_ir_subscribed:
                                        sensor_ctrl.unsub_distance()
                                        is_ir_subscribed = False

                                    print("发现目标，准备执行抓取动作...")
                                    arm_ctrl.moveto(x=250, y=80).wait_for_completed()
                                    time.sleep(1.5)
                                    chassis_ctrl.move(x=0.01, y=0, z=0, xy_speed=0.15).wait_for_completed()
                                    gripper_ctrl.close(power=50)
                                    time.sleep(1)
                                    print("物体已抓取，调整至运输姿态...")
                                    arm_ctrl.moveto(x=150, y=120).wait_for_completed()
                                    print(f"执行侧向旋转 {TRANSPORT_TURN_ANGLE} 度以清空视野...")
                                    chassis_ctrl.move(z=TRANSPORT_TURN_ANGLE, z_speed=45).wait_for_completed()
                                    bottle_is_secured = True
                                    print("状态切换：进入运输模式，开始寻找目标点。")

            # ========================= 状态二: 运输并放置瓶子 =========================
            elif bottle_is_secured:
                if not is_marker_subscribed:
                    vision_ctrl.sub_detect_info(name="marker", callback=vision_marker_update)
                    is_marker_subscribed = True
                    time.sleep(1)

                found_marker = None
                with g_marker_lock:
                    if g_detected_markers:
                        found_marker = next((m for m in g_detected_markers if m.info.isdigit()), None)

                if found_marker:
                    last_marker_timestamp = time.time()
                    tracked_marker = found_marker

                if tracked_marker and time.time() - last_marker_timestamp < MARKER_LOST_DURATION:
                    # 追踪逻辑
                    cv2.rectangle(display_frame, tracked_marker.corner1, tracked_marker.corner2, (0, 255, 0), 3)
                    cv2.putText(display_frame, f"W: {tracked_marker.width:.3f}",
                                (tracked_marker.corner1[0], tracked_marker.corner1[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 2)

                    offset_from_center = (tracked_marker.center_x - 0.5) * cam_w
                    if abs(offset_from_center) > PIXEL_ALIGN_TOLERANCE:
                        chassis_ctrl.drive_speed(x=0, y=0, z=(offset_from_center / cam_w * 80))
                        arrival_confirm_counter = 0
                        print(f"校准方向... 像素偏移: {offset_from_center:.1f}")
                    else:  # 已对准
                        chassis_ctrl.drive_speed(x=0, y=0, z=0)
                        if tracked_marker.width < MARKER_WIDTH_AT_TARGET:
                            chassis_ctrl.drive_speed(x=0.1, y=0, z=0)
                            arrival_confirm_counter = 0
                            print("方向已对准，正在接近目标点...")
                        else:  # 距离已到达
                            arrival_confirm_counter += 1
                            print(f"已到达指定区域，正在确认... [{arrival_confirm_counter}/{CONFIRM_ARRIVAL_COUNT}]")
                            if arrival_confirm_counter >= CONFIRM_ARRIVAL_COUNT:
                                chassis_ctrl.drive_speed(x=0, y=0, z=0)
                                print("已抵达目标点。")
                                print(f"正在反向旋转 {-TRANSPORT_TURN_ANGLE} 度以准备放置...")
                                chassis_ctrl.move(z=-TRANSPORT_TURN_ANGLE, z_speed=45).wait_for_completed()
                                time.sleep(0.5)

                                # --- 核心放置流程 ---
                                print("执行放置程序...")
                                arm_ctrl.moveto(x=200, y=50).wait_for_completed()
                                time.sleep(0.5)
                                gripper_ctrl.open(power=50)
                                time.sleep(1)
                                print("垂直抬升机械臂...")
                                arm_ctrl.moveto(x=200, y=50).wait_for_completed()
                                print("机械臂归位...")
                                arm_ctrl.moveto(x=180, y=90).wait_for_completed()
                                # --- 流程结束 ---

                                placement_count += 1
                                print(f"成功放置第 {placement_count} 个物体！")
                                print("向后移动，脱离放置区域...")
                                chassis_ctrl.move(x=-0.25, y=0, z=0, xy_speed=0.3).wait_for_completed()

                                print("重置任务状态，准备下一个循环...")
                                bottle_is_secured = False
                                tracked_marker = None
                                last_marker_timestamp = 0
                                arrival_confirm_counter = 0
                                g_ir_distance = None
                                gripper_ctrl.open(power=50)
                                if is_marker_subscribed:
                                    vision_ctrl.unsub_detect_info(name="marker")
                                    is_marker_subscribed = False
                                if placement_count < 5:
                                    print("开始旋转搜索下一个目标...")
                                    chassis_ctrl.drive_speed(x=0, y=0, z=15)
                                    time.sleep(0.5)
                else:  # 丢失目标
                    chassis_ctrl.drive_speed(x=0, y=0, z=12)  # 旋转寻找
                    arrival_confirm_counter = 0
                    print("视觉标签丢失，正在原地旋转搜索...")

            cv2.imshow("RoboMaster Task", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("接收到退出指令。")
                break

        if placement_count >= 2:
            print("全部任务执行完毕！")

    except Exception as e:
        print(f"\n程序运行期间出现严重错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("正在清理资源并关闭程序...")
        is_thread_active = False
        if capture_thread.is_alive():
            capture_thread.join(timeout=0.5)
        camera_ctrl.stop_video_stream()
        if is_ir_subscribed: sensor_ctrl.unsub_distance()
        if is_marker_subscribed: vision_ctrl.unsub_detect_info(name="marker")
        chassis_ctrl.drive_speed(x=0, y=0, z=0)  # 确保机器人停止
        master_robot.close()
        cv2.destroyAllWindows()
        print("程序已安全退出。")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--cam_w", type=int, default=1280, help="Width of the camera frame for processing.")
    cli_parser.add_argument("--cam_h", type=int, default=720, help="Height of the camera frame for processing.")
    cli_parser.add_argument("--model_w", type=int, default=480, help="Width of the input image for the YOLO model.")
    cli_parser.add_argument("--model_h", type=int, default=480, help="Height of the input image for the YOLO model.")
    cli_parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold for YOLO detection.")
    cli_parser.add_argument("--res", type=str, default='720p', help="Camera stream resolution ('720p' or '1080p').")
    args = cli_parser.parse_args()

    main(cam_w=args.cam_w, cam_h=args.cam_h, model_w=args.model_w, model_h=args.model_h, conf_thresh=args.conf,
         video_res=args.res)