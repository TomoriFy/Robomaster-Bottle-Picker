import argparse
import cv2
import time
import threading
import torch
from collections import deque
from robomaster import robot
from ultralytics import YOLO

# ============================================================
# 全局配置参数
# ============================================================
TRANSPORT_TURN_ANGLE = 45
CONFIRM_ARRIVAL_COUNT = 5
PIXEL_ALIGN_TOLERANCE = 35
MARKER_DIRECT_DROP_WIDTH = 0.12
MARKER_DIRECT_DROP_CONFIRM_FRAMES = 3
MARKER_DIRECT_FORWARD_M = 0.40
MARKER_DIRECT_FORWARD_SPEED = 0.30
ENABLE_MARKER_DIRECT_DROP = False  # 关闭快速放下，避免误触发
MARKER_APPROACH_TIMEOUT_S = 10.0    # 看到 marker 后持续前进这么久就进入放置
MARKER_PRE_DROP_FORWARD_M = 0.25    # 放置前再前进一小段
MARKER_PRE_DROP_FORWARD_SPEED = 0.22

# ---------- 新增：夹取瓶子后、识别 marker 前的固定运输动作 ----------
# 顺序：原地掉头 180 度 -> 前进 3 米 -> 按识别瓶子后累计转过的角度 x 再旋转 -> 再前进 3 米
POST_GRAB_TURN_BACK_ANGLE = 140
POST_GRAB_FORWARD_DISTANCE_M = 2.0
POST_GRAB_FORWARD_SPEED = 0.40
POST_GRAB_ROTATE_Z_SPEED = 45

# ---------- 新增：开场固定行驶动作 ----------
# 程序启动后：直线前进 2 米 -> 原地旋转 360 度 -> 再直线前进 2 米
# 完成后再进入原本的视觉识别 / 红外 / 抓取逻辑
STARTUP_FORWARD_DISTANCE_M = 2.0
STARTUP_FORWARD_SPEED = 0.40
STARTUP_ROTATE_ANGLE_DEG = -360
STARTUP_ROTATE_Z_SPEED = 18
STARTUP_STEP_PAUSE = 0.3

# ---------- 360 度搜索参数 ----------
INITIAL_SEARCH_Z_SPEED = 18        # 开局自身旋转搜索速度，太快就调小，比如 12
INITIAL_SEARCH_TOTAL_DEG = 360     # 每次完整搜索转 360 度
INITIAL_SEARCH_MAX_TIME = 24.0     # 最多搜索多少秒，防止死循环

# ---------- 瓶子追踪保护参数 ----------
BOTTLE_ALIGN_TOLERANCE = 35        # 瓶子中心和画面中心允许偏差
BOTTLE_TURN_GAIN = 90              # 对准瓶子时旋转比例
BOTTLE_TURN_MAX_SPEED = 25         # 对准瓶子最大旋转速度
BOTTLE_TURN_SIGN = 1.0             # 如果转反了，改成 -1.0
BOTTLE_LOST_MAX_FRAMES = 20        # 看到瓶子后短暂丢框，连续丢这么多帧才重新 360 搜索

# ---------- 红外直接夹取逻辑 ----------
IR_DIRECT_TRIGGER_DISTANCE = 200   # 红外距离 <= 200mm 后，不再视觉识别，直接按当前距离前进
IR_DIRECT_MOVE_SPEED = 0.15        # 一次性前进速度
IR_DIRECT_MOVE_OFFSET_M = 0.01     # 少走一点距离，单位 m；撞瓶子就改 0.02 或 0.03
IR_TEST_SECONDS = 5.0              # 正常任务启动时，先测试红外几秒

# ---------- 无红外备用视觉靠近参数 ----------
BOTTLE_FORWARD_SPEED_NO_IR = 0.055
VISUAL_GRAB_HEIGHT_RATIO = 0.55
MAX_VISUAL_APPROACH_TIME = 7.0

# ---------- 全局回调变量 ----------
g_ir_distance = None
g_ir_lock = threading.Lock()
g_detected_markers = []
g_marker_lock = threading.Lock()
g_cam_width, g_cam_height = 0, 0


# ============================================================
# 回调函数
# ============================================================
def infrared_sensor_update(sub_info):
    """红外距离传感器回调。一般 sub_info[0] 是距离，单位 mm。"""
    global g_ir_distance
    try:
        distance = sub_info[0]
        with g_ir_lock:
            g_ir_distance = distance if distance and distance > 0 else None
    except Exception:
        with g_ir_lock:
            g_ir_distance = None


def get_ir_distance():
    """安全读取当前红外距离。"""
    with g_ir_lock:
        return g_ir_distance


class VisionMarker:
    """视觉标签 Marker 信息。"""

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
    """视觉标签检测回调。"""
    global g_detected_markers
    with g_marker_lock:
        g_detected_markers.clear()
        for data_item in marker_data:
            x, y, w, h, info = data_item
            g_detected_markers.append(VisionMarker(x, y, w, h, info))


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


# ============================================================
# 红外测试函数
# ============================================================
def test_ir_sensor_only(test_seconds=10.0):
    """单独测试红外是否正常工作。运行：python code91_full_with_ir_test.py --test_ir"""
    global g_ir_distance

    print("正在连接机器人，准备测试红外距离传感器...")
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    sensor_ctrl = ep_robot.sensor

    g_ir_distance = None
    subscribed = False

    try:
        print("注意：当前 SDK 没有 set_distance_sensor 也没关系，直接 sub_distance 即可。")
        print("正在订阅红外距离数据...")
        sensor_ctrl.sub_distance(freq=5, callback=infrared_sensor_update)
        subscribed = True
        print("订阅成功。请把手或瓶子放到红外传感器前方，观察下面是否有 mm 数值。")

        start_time = time.time()
        got_valid_data = False

        while time.time() - start_time < test_seconds:
            ir = get_ir_distance()
            if ir is None:
                print("IR = None，暂时没有读到有效距离")
            else:
                got_valid_data = True
                print(f"IR = {ir} mm")
            time.sleep(0.2)

        if got_valid_data:
            print("红外测试结果：正常，已经读到距离数据。")
        else:
            print("红外测试结果：没有读到有效数据。")
            print("可能原因：传感器没接好、线口不对、模块没上电、距离太远、或者 SDK 订阅没有成功。")

    except Exception as e:
        print(f"红外测试失败，sub_distance 报错: {e}")
        print("如果这里报错，说明不是 set_distance_sensor 的问题，而是红外订阅本身失败。")

    finally:
        if subscribed:
            try:
                sensor_ctrl.unsub_distance()
            except Exception:
                pass
        try:
            ep_robot.close()
        except Exception:
            pass
        print("红外测试结束。")



# ============================================================
# 新增：开场固定动作函数
# ============================================================
def run_startup_motion_sequence(chassis_ctrl):
    """
    开场固定动作：
    1. 直线前进 2 米
    2. 原地旋转 360 度
    3. 再直线前进 2 米
    然后才进入原本代码的识别、靠近、夹取、运输逻辑。
    """
    print("开始执行新增开场动作：前进 2 米 -> 原地旋转 360 度 -> 再前进 2 米。")

    try:
        chassis_ctrl.drive_speed(x=0, y=0, z=0)
        time.sleep(STARTUP_STEP_PAUSE)

        print(f"开场动作 1/3：直线前进 {STARTUP_FORWARD_DISTANCE_M:.2f} m。")
        chassis_ctrl.move(
            x=STARTUP_FORWARD_DISTANCE_M,
            y=0,
            z=0,
            xy_speed=STARTUP_FORWARD_SPEED
        ).wait_for_completed()
        time.sleep(STARTUP_STEP_PAUSE)

        print(f"开场动作 2/3：原地旋转 {STARTUP_ROTATE_ANGLE_DEG} 度。")
        chassis_ctrl.drive_speed(x=0, y=0, z=0)
        time.sleep(3)
        chassis_ctrl.move(
            x=0,
            y=0,
            z=STARTUP_ROTATE_ANGLE_DEG,
            z_speed=STARTUP_ROTATE_Z_SPEED
        ).wait_for_completed()
        time.sleep(STARTUP_STEP_PAUSE)

        print(f"开场动作 3/3：再次直线前进 {STARTUP_FORWARD_DISTANCE_M:.2f} m。")
        chassis_ctrl.move(
            x=STARTUP_FORWARD_DISTANCE_M,
            y=0,
            z=0,
            xy_speed=STARTUP_FORWARD_SPEED
        ).wait_for_completed()
        time.sleep(STARTUP_STEP_PAUSE)

        chassis_ctrl.drive_speed(x=0, y=0, z=0)
        print("新增开场动作完成，开始进入原本代码逻辑。")

    except Exception as e:
        chassis_ctrl.drive_speed(x=0, y=0, z=0)
        print(f"新增开场动作执行失败，已停止底盘，错误信息: {e}")
        raise


# ============================================================
# 主程序
# ============================================================
def main(cam_w, cam_h, model_w, model_h, conf_thresh, video_res, skip_ir_start_test=False):
    global g_ir_distance, g_detected_markers, g_cam_width, g_cam_height
    g_cam_width, g_cam_height = cam_w, cam_h

    def render_detections(image, detections, class_names, m_w, m_h, c_w, c_h):
        display_img = image.copy()
        if not detections or not hasattr(detections, "boxes") or len(detections.boxes) == 0:
            return display_img

        for box in detections.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            scale_x = c_w / m_w
            scale_y = c_h / m_h
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[class_id]} {confidence:.2f}"
            cv2.putText(display_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return display_img

    def run_startup_ir_test(sensor_ctrl, seconds=IR_TEST_SECONDS):
        """正常任务启动前短测红外。不会因为测不到就退出，只打印结果。"""
        print(f"开始红外启动测试，持续 {seconds:.1f} 秒。")
        start = time.time()
        got_data = False

        while time.time() - start < seconds:
            ir = get_ir_distance()
            if ir is None:
                print("启动测试：IR = None")
            else:
                got_data = True
                print(f"启动测试：IR = {ir} mm")
            time.sleep(0.3)

        if got_data:
            print("启动测试结果：红外可以正常读数，后续会启用 ir<=200mm 直接前进夹取。")
        else:
            print("启动测试结果：暂时没读到红外数据，后续仍会继续尝试读取，但主要靠视觉。")

        return got_data

    # ---------- 初始化 YOLO ----------
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    # 按你的原代码路径保留。需要换模型就在这里改。
    model_file = r"C:\Users\27080\Desktop\final\pt\best_far.pt"

    print(f"正在从 {model_file} 加载 YOLO 模型...")
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
        print("致命错误：模型类别中没有找到 bottle。请检查 model.pt 的类别名称。")
        return
    else:
        print(f"成功找到 bottle 类别 ID: {bottle_class_id}")

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

    # ---------- 初始姿态 ----------
    print("移动机械臂至初始搜索姿态...")
    arm_ctrl.moveto(x=180, y=90).wait_for_completed()
    gripper_ctrl.open(power=50)
    print("已移动至初始搜索姿态。")

    # ---------- 新增：先执行固定开场路线 ----------
    run_startup_motion_sequence(chassis_ctrl)

    # ---------- 开启视频流 ----------
    time.sleep(2)
    print("正在开启摄像头视频流...")
    camera_ctrl.start_video_stream(display=False, resolution=video_res)
    print("视频流已成功启动。")

    frame_buffer = deque(maxlen=1)
    is_thread_active = True

    def capture_frame_loop():
        while is_thread_active:
            try:
                img = camera_ctrl.read_cv2_image(strategy="newest", timeout=1.0)
                if img is not None:
                    frame_buffer.append(img)
            except Exception as e:
                print(f"读取摄像头画面失败: {e}")
                time.sleep(0.05)

    capture_thread = threading.Thread(target=capture_frame_loop, daemon=True)
    capture_thread.start()

    # ---------- 状态变量 ----------
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
    marker_direct_confirm_counter = 0
    marker_approach_start_time = None

    # 开局 360 搜索状态
    initial_search_active = True
    initial_search_start_time = None
    initial_search_angle_est = 0.0

    # 瓶子锁定保护状态
    bottle_tracking_active = False
    bottle_lost_frames = 0
    visual_approach_start_time = None
    direct_ir_grab_running = False

    # 记录“识别到瓶子之后，为了对准瓶子实际转过的角度”
    # 单位：度；带正负号，正负方向跟 chassis_ctrl.drive_speed 的 z 方向一致。
    bottle_detect_turn_angle = 0.0
    last_bottle_turn_update_time = None

    def subscribe_ir_if_needed():
        nonlocal is_ir_subscribed
        if not is_ir_subscribed:
            try:
                # 不要调用 set_distance_sensor；当前 SDK 里这个函数可能不存在。
                sensor_ctrl.sub_distance(freq=5, callback=infrared_sensor_update)
                is_ir_subscribed = True
                print("已订阅红外距离传感器。")
            except Exception as e:
                print(f"红外订阅失败: {e}")
                is_ir_subscribed = False

    def do_grab_action():
        """执行夹取动作。"""
        nonlocal is_ir_subscribed, bottle_is_secured, visual_approach_start_time
        nonlocal initial_search_active, bottle_tracking_active, bottle_lost_frames
        nonlocal bottle_detect_turn_angle, last_bottle_turn_update_time

        chassis_ctrl.drive_speed(x=0, y=0, z=0)

        if is_ir_subscribed:
            try:
                sensor_ctrl.unsub_distance()
            except Exception:
                pass
            is_ir_subscribed = False

        print("发现目标，准备执行抓取动作...")

        # 夹取高度，如果夹不到瓶子，把 y=80 改低，比如 70 或 60
        arm_ctrl.moveto(x=250, y=80).wait_for_completed()
        time.sleep(1.2)

        # 最后补一点点距离
        chassis_ctrl.move(x=0.01, y=0, z=0, xy_speed=0.15).wait_for_completed()

        gripper_ctrl.close(power=50)
        time.sleep(1)

        print("物体已抓取，调整至运输姿态...")
        arm_ctrl.moveto(x=150, y=120).wait_for_completed()

        print(f"执行侧向旋转 {TRANSPORT_TURN_ANGLE} 度以清空视野...")
        chassis_ctrl.move(z=TRANSPORT_TURN_ANGLE, z_speed=45).wait_for_completed()

        # ---------- 新增：夹取瓶子后、识别 marker 前的固定运输动作 ----------
        # x 为识别到瓶子之后，为了对准瓶子累计转过的角度。
        # 这个动作执行完之后，才会把 bottle_is_secured 置为 True，进入 marker 识别流程。
        last_bottle_turn_update_time = None
        recorded_turn_angle = bottle_detect_turn_angle*2
        print(
            f"夹取后运输动作：先原地掉头 {POST_GRAB_TURN_BACK_ANGLE} 度，"
            f"前进 {POST_GRAB_FORWARD_DISTANCE_M:.1f} m，"
            f"再旋转记录角度 x={recorded_turn_angle:.1f} 度，"
            f"再前进 {POST_GRAB_FORWARD_DISTANCE_M:.1f} m。"
        )

        chassis_ctrl.move(
            x=0,
            y=0,
            z=POST_GRAB_TURN_BACK_ANGLE,
            z_speed=POST_GRAB_ROTATE_Z_SPEED
        ).wait_for_completed()
        time.sleep(0.2)

        chassis_ctrl.move(
            x=POST_GRAB_FORWARD_DISTANCE_M,
            y=0,
            z=0,
            xy_speed=POST_GRAB_FORWARD_SPEED
        ).wait_for_completed()
        time.sleep(0.2)

        if abs(recorded_turn_angle) > 1.0:
            chassis_ctrl.move(
                x=0,
                y=0,
                z=recorded_turn_angle,
                z_speed=POST_GRAB_ROTATE_Z_SPEED
            ).wait_for_completed()
            time.sleep(0.2)
        else:
            print("记录到的对准瓶子旋转角度接近 0 度，跳过 x 角度旋转。")

        chassis_ctrl.move(
            x=POST_GRAB_FORWARD_DISTANCE_M,
            y=0,
            z=0,
            xy_speed=POST_GRAB_FORWARD_SPEED
        ).wait_for_completed()
        time.sleep(0.2)

        bottle_is_secured = True
        visual_approach_start_time = None
        initial_search_active = False
        bottle_tracking_active = False
        bottle_lost_frames = 0

        print("状态切换：进入运输模式，开始寻找目标点。")

    def do_direct_ir_move_and_grab(ir_mm):
        """红外 <= 200mm 后：按当前红外距离一次性前进，然后直接夹取。"""
        nonlocal direct_ir_grab_running

        if direct_ir_grab_running:
            return

        direct_ir_grab_running = True
        chassis_ctrl.drive_speed(x=0, y=0, z=0)
        print(f"红外距离 {ir_mm} mm <= {IR_DIRECT_TRIGGER_DISTANCE} mm，停止视觉识别，直接前进当前距离后夹取。")

        move_m = ir_mm / 1000.0 - IR_DIRECT_MOVE_OFFSET_M
        move_m = max(0.02, move_m)

        print(f"准备一次性前进 {move_m:.3f} m，然后夹取。")
        chassis_ctrl.move(x=move_m, y=0, z=0, xy_speed=IR_DIRECT_MOVE_SPEED).wait_for_completed()
        time.sleep(0.2)

        do_grab_action()
        direct_ir_grab_running = False

    # ---------- 先订阅红外，并做启动测试 ----------
    subscribe_ir_if_needed()
    if not skip_ir_start_test:
        run_startup_ir_test(sensor_ctrl, IR_TEST_SECONDS)

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

            # ============================================================
            # 状态一：搜索并抓取瓶子
            # ============================================================
            if not bottle_is_secured:
                subscribe_ir_if_needed()

                # ---------- 红外优先逻辑 ----------
                current_ir = get_ir_distance()
                if current_ir is not None:
                    print(f"当前红外距离: {current_ir} mm")
                    if current_ir <= IR_DIRECT_TRIGGER_DISTANCE:
                        do_direct_ir_move_and_grab(current_ir)
                        cv2.imshow("RoboMaster Task", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

                # ---------- YOLO 检测 ----------
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

                found_bottle = False
                best_bottle = None

                if (
                    latest_yolo_result
                    and hasattr(latest_yolo_result, "boxes")
                    and len(latest_yolo_result.boxes) > 0
                ):
                    best_bottle = max(
                        (
                            b for b in latest_yolo_result.boxes
                            if int(b.cls[0]) == bottle_class_id
                        ),
                        key=lambda b: float(b.conf[0]),
                        default=None
                    )

                    if best_bottle:
                        found_bottle = True
                        if not bottle_tracking_active:
                            bottle_detect_turn_angle = 0.0
                            last_bottle_turn_update_time = time.time()
                            print("首次识别到瓶子，开始记录对准瓶子的累计旋转角度 x。")
                        bottle_tracking_active = True
                        bottle_lost_frames = 0

                # ---------- 开局 360 度自身旋转搜索 ----------
                if initial_search_active and not found_bottle and not bottle_tracking_active:
                    if initial_search_start_time is None:
                        initial_search_start_time = time.time()
                        initial_search_angle_est = 0.0
                        print("开局开始自身 360 度旋转搜索瓶子...")

                    elapsed = time.time() - initial_search_start_time
                    initial_search_angle_est = elapsed * INITIAL_SEARCH_Z_SPEED

                    if initial_search_angle_est < INITIAL_SEARCH_TOTAL_DEG and elapsed < INITIAL_SEARCH_MAX_TIME:
                        chassis_ctrl.drive_speed(x=0, y=0, z=INITIAL_SEARCH_Z_SPEED)
                        print(f"正在 360 搜索瓶子... 估计已转 {initial_search_angle_est:.0f}/360 度")
                    else:
                        chassis_ctrl.drive_speed(x=0, y=0, z=0)
                        initial_search_active = False
                        print("已经完成一次 360 度搜索，没有发现瓶子。继续等待视觉识别。")

                    cv2.imshow("RoboMaster Task", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("接收到退出指令。")
                        break
                    continue

                # ---------- 找到瓶子：停止搜索，进入对准 / 靠近 ----------
                if found_bottle and best_bottle is not None:
                    if initial_search_active:
                        print("360 搜索过程中发现瓶子，停止旋转，进入对准流程。")
                    initial_search_active = False
                    initial_search_start_time = None
                    chassis_ctrl.drive_speed(x=0, y=0, z=0)

                    x1, y1, x2, y2 = best_bottle.xyxy[0].cpu().numpy()
                    center_x_px = ((x1 + x2) / 2) * (cam_w / model_w)
                    offset_px = center_x_px - (cam_w / 2)
                    box_height_ratio = (y2 - y1) / model_h
                    confidence = float(best_bottle.conf[0])

                    print(
                        f"检测到瓶子 conf={confidence:.2f}, "
                        f"offset={offset_px:.1f}, box_h={box_height_ratio:.2f}, ir={current_ir}"
                    )

                    # 还没对准，先旋转对准
                    if abs(offset_px) > BOTTLE_ALIGN_TOLERANCE:
                        visual_approach_start_time = None
                        z_speed = (offset_px / cam_w) * BOTTLE_TURN_GAIN
                        z_speed = clamp(z_speed, -BOTTLE_TURN_MAX_SPEED, BOTTLE_TURN_MAX_SPEED)
                        z_speed = z_speed * BOTTLE_TURN_SIGN

                        now = time.time()
                        if last_bottle_turn_update_time is not None:
                            dt = now - last_bottle_turn_update_time
                            if dt > 0:
                                bottle_detect_turn_angle += z_speed * dt
                        last_bottle_turn_update_time = now

                        chassis_ctrl.drive_speed(x=0, y=0, z=z_speed)
                        print(
                            f"正在对准瓶子，旋转速度 z={z_speed:.1f}，"
                            f"已累计记录角度 x={bottle_detect_turn_angle:.1f} 度"
                        )

                    # 已经对准，但红外还没到 200，先视觉慢速靠近
                    else:
                        last_bottle_turn_update_time = None

                        if visual_approach_start_time is None:
                            visual_approach_start_time = time.time()

                        visual_time = time.time() - visual_approach_start_time

                        if box_height_ratio >= VISUAL_GRAB_HEIGHT_RATIO and current_ir is None:
                            print("红外无数据，但视觉判断已经很近，尝试夹取。")
                            do_grab_action()
                        elif visual_time >= MAX_VISUAL_APPROACH_TIME and current_ir is None:
                            print("红外无数据，视觉靠近时间已到，尝试夹取。")
                            do_grab_action()
                        else:
                            chassis_ctrl.drive_speed(x=BOTTLE_FORWARD_SPEED_NO_IR, y=0, z=0)
                            print("已对准瓶子，红外还未触发 <=200，先慢速前进靠近。")

                # ---------- 看到过瓶子后短暂丢框：保护机制 ----------
                else:
                    if bottle_tracking_active:
                        bottle_lost_frames += 1

                        if bottle_lost_frames <= BOTTLE_LOST_MAX_FRAMES:
                            chassis_ctrl.drive_speed(x=0, y=0, z=0)
                            print(
                                f"瓶子短暂丢失，先停住等待重新识别... "
                                f"[{bottle_lost_frames}/{BOTTLE_LOST_MAX_FRAMES}]"
                            )
                        else:
                            print("瓶子连续丢失太久，重新进入 360 度搜索。")
                            bottle_tracking_active = False
                            bottle_lost_frames = 0
                            visual_approach_start_time = None
                            bottle_detect_turn_angle = 0.0
                            last_bottle_turn_update_time = None
                            initial_search_active = True
                            initial_search_start_time = None
                            chassis_ctrl.drive_speed(x=0, y=0, z=0)
                    else:
                        # 没有锁定过瓶子，继续 360 搜索
                        initial_search_active = True

            # ============================================================
            # 状态二：运输并放置瓶子
            # ============================================================
            else:
                if not is_marker_subscribed:
                    try:
                        vision_ctrl.sub_detect_info(name="marker", callback=vision_marker_update)
                        is_marker_subscribed = True
                        time.sleep(1)
                        print("已订阅视觉标签检测。")
                    except Exception as e:
                        print(f"订阅视觉标签失败: {e}")

                found_marker = None
                with g_marker_lock:
                    if g_detected_markers:
                        found_marker = next((m for m in g_detected_markers if m.info == '3'), None)

                if found_marker:
                    last_marker_timestamp = time.time()
                    tracked_marker = found_marker

                if tracked_marker and time.time() - last_marker_timestamp < MARKER_LOST_DURATION:
                    cv2.rectangle(display_frame, tracked_marker.corner1, tracked_marker.corner2, (0, 255, 0), 3)
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

                    # 新增：识别到 marker 后，若宽度较小且已对准中心，连续确认后直接前进放下
                    direct_drop_ready = (
                        tracked_marker.width < MARKER_DIRECT_DROP_WIDTH
                        and abs(offset_from_center) <= PIXEL_ALIGN_TOLERANCE
                    )
                    if direct_drop_ready:
                        marker_direct_confirm_counter += 1
                        print(
                            f"快速放下条件确认 "
                            f"[{marker_direct_confirm_counter}/{MARKER_DIRECT_DROP_CONFIRM_FRAMES}]，"
                            f"w={tracked_marker.width:.3f}, offset={offset_from_center:.1f}"
                        )
                    else:
                        marker_direct_confirm_counter = 0

                    if ENABLE_MARKER_DIRECT_DROP and marker_direct_confirm_counter >= MARKER_DIRECT_DROP_CONFIRM_FRAMES:
                        chassis_ctrl.drive_speed(x=0, y=0, z=0)
                        print(f"检测到 marker 宽度 {tracked_marker.width:.3f} < {MARKER_DIRECT_DROP_WIDTH:.2f}，"
                              f"前进 {MARKER_DIRECT_FORWARD_M:.2f}m 直接放下。")
                        chassis_ctrl.move(
                            x=MARKER_DIRECT_FORWARD_M,
                            y=0,
                            z=0,
                            xy_speed=MARKER_DIRECT_FORWARD_SPEED
                        ).wait_for_completed()

                        print("执行放置程序...")
                        arm_ctrl.moveto(x=200, y=50).wait_for_completed()
                        time.sleep(0.5)
                        gripper_ctrl.open(power=50)
                        time.sleep(1)
                        print("机械臂归位...")
                        arm_ctrl.moveto(x=180, y=90).wait_for_completed()

                        placement_count += 1
                        print(f"成功放置第 {placement_count} 个物体！")
                        print("向后移动，脱离放置区域...")
                        chassis_ctrl.move(x=-0.25, y=0, z=0, xy_speed=0.3).wait_for_completed()

                        print("重置任务状态，准备下一个循环...")
                        bottle_is_secured = False
                        tracked_marker = None
                        last_marker_timestamp = 0
                        arrival_confirm_counter = 0
                        marker_direct_confirm_counter = 0
                        g_ir_distance = None
                        visual_approach_start_time = None
                        direct_ir_grab_running = False

                        initial_search_active = True
                        initial_search_start_time = None
                        bottle_tracking_active = False
                        bottle_lost_frames = 0
                        bottle_detect_turn_angle = 0.0
                        last_bottle_turn_update_time = None

                        gripper_ctrl.open(power=50)

                        if is_marker_subscribed:
                            try:
                                vision_ctrl.unsub_detect_info(name="marker")
                            except Exception:
                                pass
                            is_marker_subscribed = False

                        cv2.imshow("RoboMaster Task", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("接收到退出指令。")
                            break
                        continue

                    if abs(offset_from_center) > PIXEL_ALIGN_TOLERANCE:
                        chassis_ctrl.drive_speed(x=0, y=0, z=(offset_from_center / cam_w * 80))
                        arrival_confirm_counter = 0
                        marker_direct_confirm_counter = 0
                        marker_approach_start_time = None
                        print(f"校准方向... 像素偏移: {offset_from_center:.1f}")
                    else:
                        chassis_ctrl.drive_speed(x=0, y=0, z=0)

                        if tracked_marker.width < MARKER_WIDTH_AT_TARGET:
                            chassis_ctrl.drive_speed(x=0.1, y=0, z=0)
                            arrival_confirm_counter = 0
                            marker_direct_confirm_counter = 0
                            if marker_approach_start_time is None:
                                marker_approach_start_time = time.time()
                            approach_elapsed = time.time() - marker_approach_start_time
                            print(f"方向已对准，正在接近目标点... 已前进 {approach_elapsed:.1f}s")

                            if approach_elapsed >= MARKER_APPROACH_TIMEOUT_S:
                                chassis_ctrl.drive_speed(x=0, y=0, z=0)
                                print(f"接近超时达到 {MARKER_APPROACH_TIMEOUT_S:.1f}s，进入放置阶段。")
                                print(f"放置前再前进 {MARKER_PRE_DROP_FORWARD_M:.2f} m。")
                                chassis_ctrl.move(
                                    x=MARKER_PRE_DROP_FORWARD_M,
                                    y=0,
                                    z=0,
                                    xy_speed=MARKER_PRE_DROP_FORWARD_SPEED
                                ).wait_for_completed()
                                time.sleep(0.2)

                                print("执行放置程序...")
                                arm_ctrl.moveto(x=200, y=50).wait_for_completed()
                                time.sleep(0.5)
                                gripper_ctrl.open(power=50)
                                time.sleep(1)
                                print("机械臂归位...")
                                arm_ctrl.moveto(x=180, y=90).wait_for_completed()

                                placement_count += 1
                                print(f"成功放置第 {placement_count} 个物体！")
                                print("向后移动，脱离放置区域...")
                                chassis_ctrl.move(x=-0.25, y=0, z=0, xy_speed=0.3).wait_for_completed()

                                print("重置任务状态，准备下一个循环...")
                                bottle_is_secured = False
                                tracked_marker = None
                                last_marker_timestamp = 0
                                arrival_confirm_counter = 0
                                marker_direct_confirm_counter = 0
                                marker_approach_start_time = None
                                g_ir_distance = None
                                visual_approach_start_time = None
                                direct_ir_grab_running = False

                                initial_search_active = True
                                initial_search_start_time = None
                                bottle_tracking_active = False
                                bottle_lost_frames = 0
                                bottle_detect_turn_angle = 0.0
                                last_bottle_turn_update_time = None

                                gripper_ctrl.open(power=50)

                                if is_marker_subscribed:
                                    try:
                                        vision_ctrl.unsub_detect_info(name="marker")
                                    except Exception:
                                        pass
                                    is_marker_subscribed = False
                        else:
                            arrival_confirm_counter += 1
                            marker_approach_start_time = None
                            print(f"已到达指定区域，正在确认... [{arrival_confirm_counter}/{CONFIRM_ARRIVAL_COUNT}]")

                            if arrival_confirm_counter >= CONFIRM_ARRIVAL_COUNT:
                                chassis_ctrl.drive_speed(x=0, y=0, z=0)
                                print("已抵达目标点。")
                                print(f"正在反向旋转 {-TRANSPORT_TURN_ANGLE} 度以准备放置...")
                                chassis_ctrl.move(z=-TRANSPORT_TURN_ANGLE, z_speed=45).wait_for_completed()
                                time.sleep(0.5)

                                print("执行放置程序...")
                                arm_ctrl.moveto(x=200, y=50).wait_for_completed()
                                time.sleep(0.5)
                                gripper_ctrl.open(power=50)
                                time.sleep(1)
                                print("机械臂归位...")
                                arm_ctrl.moveto(x=180, y=90).wait_for_completed()

                                placement_count += 1
                                print(f"成功放置第 {placement_count} 个物体！")

                                print("向后移动，脱离放置区域...")
                                chassis_ctrl.move(x=-0.25, y=0, z=0, xy_speed=0.3).wait_for_completed()

                                print("重置任务状态，准备下一个循环...")
                                bottle_is_secured = False
                                tracked_marker = None
                                last_marker_timestamp = 0
                                arrival_confirm_counter = 0
                                marker_direct_confirm_counter = 0
                                marker_approach_start_time = None
                                g_ir_distance = None
                                visual_approach_start_time = None
                                direct_ir_grab_running = False

                                initial_search_active = True
                                initial_search_start_time = None
                                bottle_tracking_active = False
                                bottle_lost_frames = 0
                                bottle_detect_turn_angle = 0.0
                                last_bottle_turn_update_time = None

                                gripper_ctrl.open(power=50)

                                if is_marker_subscribed:
                                    try:
                                        vision_ctrl.unsub_detect_info(name="marker")
                                    except Exception:
                                        pass
                                    is_marker_subscribed = False
                else:
                    # 恢复之前思路：marker 丢失时继续旋转搜索，不直接放下
                    chassis_ctrl.drive_speed(x=0, y=0, z=12)
                    arrival_confirm_counter = 0
                    marker_direct_confirm_counter = 0
                    marker_approach_start_time = None
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

        if is_ir_subscribed:
            try:
                sensor_ctrl.unsub_distance()
            except Exception:
                pass

        if is_marker_subscribed:
            try:
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

    cli_parser.add_argument("--cam_w", type=int, default=1280, help="Width of the camera frame for processing.")
    cli_parser.add_argument("--cam_h", type=int, default=720, help="Height of the camera frame for processing.")
    cli_parser.add_argument("--model_w", type=int, default=480, help="Width of the input image for the YOLO model.")
    cli_parser.add_argument("--model_h", type=int, default=480, help="Height of the input image for the YOLO model.")
    cli_parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for YOLO detection.")
    cli_parser.add_argument("--res", type=str, default="720p", help="Camera stream resolution: '720p' or '1080p'.")

    cli_parser.add_argument("--test_ir", action="store_true", help="Only test IR distance sensor and exit.")
    cli_parser.add_argument("--test_ir_seconds", type=float, default=10.0, help="IR test duration in seconds.")
    cli_parser.add_argument("--skip_ir_start_test", action="store_true", help="Skip startup IR test in normal task mode.")

    args = cli_parser.parse_args()

    if args.test_ir:
        test_ir_sensor_only(test_seconds=args.test_ir_seconds)
    else:
        main(
            cam_w=args.cam_w,
            cam_h=args.cam_h,
            model_w=args.model_w,
            model_h=args.model_h,
            conf_thresh=args.conf,
            video_res=args.res,
            skip_ir_start_test=args.skip_ir_start_test
        )
