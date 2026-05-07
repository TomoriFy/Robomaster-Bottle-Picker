import argparse
import cv2
import time
import threading
import torch
from collections import deque
from robomaster import robot
from ultralytics import YOLO


# ---------- 全局配置参数 ----------
TRANSPORT_TURN_ANGLE = 45
CONFIRM_ARRIVAL_COUNT = 5
PIXEL_ALIGN_TOLERANCE = 35

# ---------- 瓶子姿态判断与绕瓶参数 ----------
BOTTLE_VERTICAL_RATIO = 1.25
BOTTLE_HORIZONTAL_RATIO = 0.85

END_VIEW_RATIO_LOW = 0.75
END_VIEW_RATIO_HIGH = 1.35

CIRCLE_SEARCH_STEPS = 14
CIRCLE_MOVE_TIME = 0.28
CIRCLE_Y_SPEED = 0.08
CIRCLE_Z_SPEED = 10

SAFE_DISTANCE_MM = 220
MIN_GRAB_DISTANCE_MM = 60

# ---------- 全局状态 ----------
g_ir_distance = None
g_detected_markers = []
g_marker_lock = threading.Lock()
g_cam_width, g_cam_height = 0, 0


# ---------- 传感器与视觉回调函数 ----------
def infrared_sensor_update(sub_info):
    global g_ir_distance
    distance = sub_info[0]
    g_ir_distance = distance if distance and distance > 0 else None


class VisionMarker:
    def __init__(self, x, y, w, h, info_text):
        self._norm_x = x
        self._norm_w = w
        self._text = info_text

        self.corner1 = (
            int((x - w / 2) * g_cam_width),
            int((y - h / 2) * g_cam_height)
        )
        self.corner2 = (
            int((x + w / 2) * g_cam_width),
            int((y + h / 2) * g_cam_height)
        )

    @property
    def center_x(self):
        return self._norm_x

    @property
    def width(self):
        return self._norm_w

    @property
    def info(self):
        return self._text


def vision_marker_update(marker_data):
    global g_detected_markers
    with g_marker_lock:
        g_detected_markers.clear()
        for data_item in marker_data:
            x, y, w, h, info = data_item
            g_detected_markers.append(VisionMarker(x, y, w, h, info))


def main(cam_w, cam_h, model_w, model_h, conf_thresh, video_res):
    global g_ir_distance, g_detected_markers, g_cam_width, g_cam_height

    g_cam_width, g_cam_height = cam_w, cam_h

    # ---------- YOLO 绘图函数 ----------
    def render_detections(image, detections, class_names, m_w, m_h, c_w, c_h):
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
            cv2.putText(
                display_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        return display_img

    def get_best_bottle_from_result(result):
        if result is None or not hasattr(result, "boxes") or len(result.boxes) == 0:
            return None

        best_bottle = max(
            (b for b in result.boxes if int(b.cls[0]) == bottle_class_id),
            key=lambda b: float(b.conf[0]),
            default=None
        )

        return best_bottle

    def get_best_bottle_from_frame(frame):
        if frame is None:
            return None, None

        model_input_frame = cv2.resize(frame, (model_w, model_h))

        results = yolo_model(
            model_input_frame,
            conf=conf_thresh,
            device=compute_device,
            stream=False
        )

        if not results:
            return None, None

        result = results[0]
        best_bottle = get_best_bottle_from_result(result)

        return best_bottle, result

    def estimate_bottle_pose(box):
        if box is None:
            return "unknown", 0.0

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            return "unknown", 0.0

        ratio = box_h / box_w

        if ratio > BOTTLE_VERTICAL_RATIO:
            return "vertical", ratio
        elif ratio < BOTTLE_HORIZONTAL_RATIO:
            return "horizontal", ratio
        else:
            return "end_or_uncertain", ratio

    def get_bottle_center_offset(box):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x_px = ((x1 + x2) / 2) * (cam_w / model_w)
        offset_px = center_x_px - (cam_w / 2)
        return offset_px

    def draw_pose_info(display_frame, box, pose, ratio):
        if box is None:
            return display_frame

        text = f"pose: {pose}, ratio: {ratio:.2f}"
        cv2.putText(
            display_frame,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        return display_frame

    def align_to_bottle_center(box, tolerance_px=30):
        offset_px = get_bottle_center_offset(box)

        if abs(offset_px) > tolerance_px:
            z_speed = (offset_px / cam_w) * 100
            chassis_ctrl.drive_speed(x=0, y=0, z=z_speed)
            print(f"正在对准瓶子中心，offset={offset_px:.1f}, z_speed={z_speed:.1f}")
            return False

        chassis_ctrl.drive_speed(x=0, y=0, z=0)
        return True

    def circle_search_bottle_end():
        global g_ir_distance

        print("检测到瓶子横放，开始绕瓶寻找瓶底方向...")

        if g_ir_distance and g_ir_distance < SAFE_DISTANCE_MM:
            print(f"当前距离过近：{g_ir_distance}mm，先后退一点再绕瓶")
            chassis_ctrl.drive_speed(x=-0.06, y=0, z=0)
            time.sleep(0.5)
            chassis_ctrl.drive_speed(x=0, y=0, z=0)
            time.sleep(0.2)

        for direction in [1, -1]:
            print("开始向{}侧绕瓶搜索...".format("右" if direction == 1 else "左"))

            for step in range(CIRCLE_SEARCH_STEPS):
                chassis_ctrl.drive_speed(
                    x=0,
                    y=direction * CIRCLE_Y_SPEED,
                    z=direction * CIRCLE_Z_SPEED
                )
                time.sleep(CIRCLE_MOVE_TIME)

                chassis_ctrl.drive_speed(x=0, y=0, z=0)
                time.sleep(0.12)

                frame = frame_buffer[-1] if len(frame_buffer) else None
                best_bottle, result = get_best_bottle_from_frame(frame)

                if best_bottle is None:
                    print(f"绕瓶 step={step}: 暂时丢失瓶子")
                    continue

                pose, ratio = estimate_bottle_pose(best_bottle)
                offset_px = get_bottle_center_offset(best_bottle)
                confidence = float(best_bottle.conf[0])

                print(
                    f"绕瓶 step={step}, pose={pose}, ratio={ratio:.2f}, "
                    f"offset={offset_px:.1f}, conf={confidence:.2f}"
                )

                if abs(offset_px) > 100:
                    z_speed = (offset_px / cam_w) * 80
                    chassis_ctrl.drive_speed(x=0, y=0, z=z_speed)
                    time.sleep(0.18)
                    chassis_ctrl.drive_speed(x=0, y=0, z=0)

                if END_VIEW_RATIO_LOW <= ratio <= END_VIEW_RATIO_HIGH:
                    print("检测框比例接近瓶底/瓶口视角，准备进入抓取流程")
                    chassis_ctrl.drive_speed(x=0, y=0, z=0)
                    return True

            print("这一侧没有找到合适角度，回到中间附近")

            chassis_ctrl.drive_speed(
                x=0,
                y=-direction * CIRCLE_Y_SPEED,
                z=-direction * CIRCLE_Z_SPEED
            )
            time.sleep(CIRCLE_MOVE_TIME * CIRCLE_SEARCH_STEPS * 0.6)
            chassis_ctrl.drive_speed(x=0, y=0, z=0)
            time.sleep(0.3)

        print("绕瓶搜索结束，没有稳定找到瓶底方向")
        return False

    def grab_bottle_sequence():
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

    # ---------- 初始化 YOLO 模型 ----------
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    model_file = r"/home/nvidia/Desktop/Final Project/model.pt"

    print(f"正在从 {model_file} 加载YOLO模型...")
    yolo_model = YOLO(model_file)
    yolo_model.fuse()
    yolo_model.to(compute_device)

    yolo_class_names = yolo_model.names

    print(f"模型加载成功，使用的计算设备: {compute_device}")
    print("模型支持的类别:", yolo_class_names)

    bottle_class_id = -1
    for class_id, class_name in yolo_class_names.items():
        if class_name.lower() == "bottle":
            bottle_class_id = class_id
            break

    if bottle_class_id == -1:
        print("致命错误：在模型类别中未找到 'bottle'。请检查模型文件或类别名称。")
        return
    else:
        print(f"成功找到 'bottle' 对应的类别ID为: {bottle_class_id}")

    # ---------- 初始化机器人 ----------
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

    # ---------- 设置机器人初始状态 ----------
    print("移动机械臂至初始搜索姿态...")
    arm_ctrl.moveto(x=180, y=90).wait_for_completed()
    gripper_ctrl.open(power=50)
    print("已移动至初始搜索姿态。")

    # ---------- 启动视频流 ----------
    time.sleep(2)

    print("正在开启摄像头视频流...")
    camera_ctrl.start_video_stream(display=False, resolution=video_res)
    print("视频流已成功启动。")

    frame_buffer = deque(maxlen=1)
    is_thread_active = True

    def capture_frame_loop():
        while is_thread_active:
            img = camera_ctrl.read_cv2_image(strategy="newest", timeout=1.0)
            if img is not None:
                frame_buffer.append(img)

    capture_thread = threading.Thread(target=capture_frame_loop, daemon=True)
    capture_thread.start()

    # ---------- 初始化任务状态变量 ----------
    frame_process_counter = 0
    latest_yolo_result = None

    is_ir_subscribed = False
    is_marker_subscribed = False

    bottle_is_secured = False
    placement_count = 0

    last_marker_timestamp = 0
    tracked_marker = None

    MARKER_LOST_DURATION = 0.75
    MARKER_WIDTH_AT_TARGET = 0.25
    arrival_confirm_counter = 0

    try:
        sensor_ctrl.set_distance_sensor(1)
    except Exception as e:
        print(f"提醒: 红外传感器设置失败，可能影响近距离判断: {e}")

    # ---------- 主循环 ----------
    try:
        while placement_count < 2:
            current_frame = frame_buffer[-1] if len(frame_buffer) else None

            if current_frame is None:
                time.sleep(0.01)
                continue

            if current_frame.shape[1] != cam_w or current_frame.shape[0] != cam_h:
                current_frame = cv2.resize(current_frame, (cam_w, cam_h))

            display_frame = current_frame.copy()

            # ========================= 状态一：搜索并抓取瓶子 =========================
            if not bottle_is_secured:
                model_input_frame = cv2.resize(current_frame, (model_w, model_h))

                results_stream = yolo_model(
                    model_input_frame,
                    conf=conf_thresh,
                    device=compute_device,
                    stream=True
                )

                for res in results_stream:
                    latest_yolo_result = res
                    display_frame = render_detections(
                        current_frame,
                        latest_yolo_result,
                        yolo_class_names,
                        model_w,
                        model_h,
                        cam_w,
                        cam_h
                    )
                    frame_process_counter += 1
                    break

                if frame_process_counter > 20:
                    if not is_ir_subscribed:
                        sensor_ctrl.sub_distance(freq=5, callback=infrared_sensor_update)
                        is_ir_subscribed = True

                    best_bottle = get_best_bottle_from_result(latest_yolo_result)

                    if best_bottle:
                        pose, ratio = estimate_bottle_pose(best_bottle)
                        offset_px = get_bottle_center_offset(best_bottle)

                        display_frame = draw_pose_info(display_frame, best_bottle, pose, ratio)

                        print(
                            f"当前瓶子姿态判断：{pose}, "
                            f"ratio={ratio:.2f}, offset={offset_px:.1f}, "
                            f"ir={g_ir_distance}"
                        )

                        # 横放：先绕瓶找瓶底/瓶口方向
                        if pose == "horizontal":
                            chassis_ctrl.drive_speed(x=0, y=0, z=0)

                            found_end = circle_search_bottle_end()

                            if not found_end:
                                print("未找到合适瓶底方向，继续重新搜索瓶子")
                                continue

                            print("已调整到更适合夹取的方向，重新进入对准流程")
                            continue

                        # 竖放或接近瓶底方向：正常对准
                        aligned = align_to_bottle_center(best_bottle, tolerance_px=30)

                        if not aligned:
                            continue

                        if g_ir_distance and g_ir_distance > MIN_GRAB_DISTANCE_MM:
                            chassis_ctrl.drive_speed(x=0.07, y=0, z=0)
                            print(f"正在靠近瓶子，当前红外距离：{g_ir_distance}mm")

                        elif g_ir_distance:
                            chassis_ctrl.drive_speed(x=0, y=0, z=0)

                            if is_ir_subscribed:
                                sensor_ctrl.unsub_distance()
                                is_ir_subscribed = False

                            grab_bottle_sequence()

                            bottle_is_secured = True
                            print("状态切换：进入运输模式，开始寻找目标点。")

                    else:
                        chassis_ctrl.drive_speed(x=0, y=0, z=12)
                        print("未检测到瓶子，正在旋转搜索...")

            # ========================= 状态二：运输并放置瓶子 =========================
            elif bottle_is_secured:
                if not is_marker_subscribed:
                    vision_ctrl.sub_detect_info(name="marker", callback=vision_marker_update)
                    is_marker_subscribed = True
                    time.sleep(1)

                found_marker = None

                with g_marker_lock:
                    if g_detected_markers:
                        found_marker = next(
                            (m for m in g_detected_markers if m.info.isdigit()),
                            None
                        )

                if found_marker:
                    last_marker_timestamp = time.time()
                    tracked_marker = found_marker

                if tracked_marker and time.time() - last_marker_timestamp < MARKER_LOST_DURATION:
                    cv2.rectangle(
                        display_frame,
                        tracked_marker.corner1,
                        tracked_marker.corner2,
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        display_frame,
                        f"W: {tracked_marker.width:.3f}",
                        (tracked_marker.corner1[0], tracked_marker.corner1[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        2
                    )

                    offset_from_center = (tracked_marker.center_x - 0.5) * cam_w

                    if abs(offset_from_center) > PIXEL_ALIGN_TOLERANCE:
                        chassis_ctrl.drive_speed(
                            x=0,
                            y=0,
                            z=(offset_from_center / cam_w * 80)
                        )
                        arrival_confirm_counter = 0
                        print(f"校准方向... 像素偏移: {offset_from_center:.1f}")

                    else:
                        chassis_ctrl.drive_speed(x=0, y=0, z=0)

                        if tracked_marker.width < MARKER_WIDTH_AT_TARGET:
                            chassis_ctrl.drive_speed(x=0.1, y=0, z=0)
                            arrival_confirm_counter = 0
                            print("方向已对准，正在接近目标点...")

                        else:
                            arrival_confirm_counter += 1
                            print(
                                f"已到达指定区域，正在确认... "
                                f"[{arrival_confirm_counter}/{CONFIRM_ARRIVAL_COUNT}]"
                            )

                            if arrival_confirm_counter >= CONFIRM_ARRIVAL_COUNT:
                                chassis_ctrl.drive_speed(x=0, y=0, z=0)

                                print("已抵达目标点。")
                                print(f"正在反向旋转 {-TRANSPORT_TURN_ANGLE} 度以准备放置...")

                                chassis_ctrl.move(
                                    z=-TRANSPORT_TURN_ANGLE,
                                    z_speed=45
                                ).wait_for_completed()

                                time.sleep(0.5)

                                print("执行放置程序...")

                                arm_ctrl.moveto(x=200, y=50).wait_for_completed()
                                time.sleep(0.5)

                                gripper_ctrl.open(power=50)
                                time.sleep(1)

                                print("垂直抬升机械臂...")
                                arm_ctrl.moveto(x=200, y=50).wait_for_completed()

                                print("机械臂归位...")
                                arm_ctrl.moveto(x=180, y=90).wait_for_completed()

                                placement_count += 1
                                print(f"成功放置第 {placement_count} 个物体！")

                                print("向后移动，脱离放置区域...")
                                chassis_ctrl.move(
                                    x=-0.25,
                                    y=0,
                                    z=0,
                                    xy_speed=0.3
                                ).wait_for_completed()

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

                                if placement_count < 2:
                                    print("开始旋转搜索下一个目标...")
                                    chassis_ctrl.drive_speed(x=0, y=0, z=15)
                                    time.sleep(0.5)

                else:
                    chassis_ctrl.drive_speed(x=0, y=0, z=12)
                    arrival_confirm_counter = 0
                    print("视觉标签丢失，正在原地旋转搜索...")

            cv2.imshow("RoboMaster Task", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
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

        try:
            camera_ctrl.stop_video_stream()
        except Exception:
            pass

        try:
            if is_ir_subscribed:
                sensor_ctrl.unsub_distance()
        except Exception:
            pass

        try:
            if is_marker_subscribed:
                vision_ctrl.unsub_detect_info(name="marker")
        except Exception:
            pass

        try:
            chassis_ctrl.drive_speed(x=0, y=0, z=0)
        except Exception:
            pass

        try:
            master_robot.close()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("程序已安全退出。")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument(
        "--cam_w",
        type=int,
        default=1280,
        help="Width of the camera frame for processing."
    )

    cli_parser.add_argument(
        "--cam_h",
        type=int,
        default=720,
        help="Height of the camera frame for processing."
    )

    cli_parser.add_argument(
        "--model_w",
        type=int,
        default=480,
        help="Width of the input image for the YOLO model."
    )

    cli_parser.add_argument(
        "--model_h",
        type=int,
        default=480,
        help="Height of the input image for the YOLO model."
    )

    cli_parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="Confidence threshold for YOLO detection."
    )

    cli_parser.add_argument(
        "--res",
        type=str,
        default="720p",
        help="Camera stream resolution, such as '720p' or '1080p'."
    )

    args = cli_parser.parse_args()

    main(
        cam_w=args.cam_w,
        cam_h=args.cam_h,
        model_w=args.model_w,
        model_h=args.model_h,
        conf_thresh=args.conf,
        video_res=args.res
    )
