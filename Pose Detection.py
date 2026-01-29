##########################################################################
# Picamera2 Capture + Display (Raspberry Pi)
#
# - Uses the Picamera2 threaded wrapper
# - Displays frames via OpenCV (no analysis)
# - Throttles display rate in THIS demo (consumer-side)
##########################################################################

from __future__ import annotations

import logging
import os
import time
import cv2
import sys
import numpy as np
from typing import Tuple, List

# Add Data directory to path for local model modules
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname("/home/agr5/Downloads/camera/code/data"), "Data"))
if DATA_DIR not in sys.path:
    sys.path.append(DATA_DIR)

from mp_persondet import MPPersonDet
from mp_pose import MPPose

MODEL_FILES = {
    "pose": "pose_estimation_mediapipe_2023mar.onnx",
    "person": "person_detection_mediapipe_2023mar.onnx",
}

def model_path(model_key: str) -> str:
    return os.path.join(DATA_DIR, MODEL_FILES[model_key])


def load_models(logger: logging.Logger) -> Tuple[MPPersonDet, MPPose]:
    """Loads person detection and pose estimation models."""
    pose_path = model_path("pose")
    person_path = model_path("person")

    if not os.path.exists(pose_path):
        logger.log(logging.CRITICAL, "Model file not found: %s", pose_path)
        raise SystemExit(1)
    if not os.path.exists(person_path):
        logger.log(logging.CRITICAL, "Model file not found: %s", person_path)
        raise SystemExit(1)

    backend_id = cv2.dnn.DNN_BACKEND_OPENCV
    target_id = cv2.dnn.DNN_TARGET_CPU
    person_detector = MPPersonDet(
        modelPath=person_path,
        nmsThreshold=0.3,
        scoreThreshold=0.5,
        topK=5000,
        backendId=backend_id,
        targetId=target_id,
    )
    pose_estimator = MPPose(
        modelPath=pose_path,
        confThreshold=0.8,
        backendId=backend_id,
        targetId=target_id,
    )
    return person_detector, pose_estimator


def visualize(
    image: np.ndarray,
    poses: List,
    *,
    draw_3d: bool = False,
    draw_mask_edges: bool = False,
    line_thickness: int = 1,
    point_radius: int = 1,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Draws 2D and 3D pose visualizations on images.

    Args:
        image (np.ndarray): The input image (BGR).
        poses (List): List of pose results, each containing bounding box, landmarks, mask, etc.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (2D visualization image, 3D visualization image)
    """
    # Constants for display sizes and colors
    DISPLAY_SIZE = 400
    DISPLAY_CENTER = 200
    SCALE = 100
    COLOR_WHITE = (255, 255, 255)
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)

    def draw_skeleton_lines(canvas, landmarks, keep_landmarks, thickness: int):
        """Draws skeleton lines between keypoints if both are present."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (12, 14), (14, 16), (16, 22), (16, 18), (16, 20), (18, 20),
            (11, 13), (13, 15), (15, 21), (15, 19), (15, 17), (17, 19),
            (11, 12), (11, 23), (23, 24), (24, 12), (24, 26), (26, 28),
            (28, 30), (28, 32), (30, 32), (23, 25), (25, 27), (27, 31),
            (27, 29), (29, 31)
        ]
        for idx1, idx2 in connections:
            if keep_landmarks[idx1] and keep_landmarks[idx2]:
                cv2.line(canvas, landmarks[idx1], landmarks[idx2], COLOR_WHITE, thickness)

    def draw_keypoints(canvas, landmarks, keep_landmarks, color=COLOR_RED, radius: int = 1):
        """Draws keypoints on the canvas."""
        for i, point in enumerate(landmarks):
            if keep_landmarks[i]:
                cv2.circle(canvas, tuple(point), radius, color, -1)

    def draw_edges_on_mask(mask, display_screen):
        """Draws green edges from mask onto the display image."""
        edges = cv2.Canny(mask, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[edges == 255] = COLOR_GREEN
        return cv2.add(edges_bgr, display_screen)

    def draw_pose_2d(display_screen, bbox, conf, landmarks_screen, keep_landmarks):
        """Draws 2D pose: bounding box, confidence, skeleton, and keypoints."""
        bbox = bbox.astype(np.int32)
        cv2.rectangle(display_screen, tuple(bbox[0]), tuple(bbox[1]), COLOR_GREEN, max(1, line_thickness))
        cv2.putText(
            display_screen,
            "{:.4f}".format(conf),
            (bbox[0][0], bbox[0][1] + 12),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            COLOR_RED,
        )
        landmarks_xy = landmarks_screen[:, 0:2].astype(np.int32)
        draw_skeleton_lines(display_screen, landmarks_xy, keep_landmarks, thickness=max(1, line_thickness))
        draw_keypoints(display_screen, landmarks_xy, keep_landmarks, color=COLOR_RED, radius=max(1, point_radius))

    def draw_pose_3d(display_3d, landmarks_world, keep_landmarks):
        """Draws 3D projections of the pose on the display_3d canvas."""
        # Main View (XY)
        landmarks_xy = (landmarks_world[:, [0, 1]] * SCALE + DISPLAY_CENTER).astype(np.int32)
        draw_skeleton_lines(display_3d, landmarks_xy, keep_landmarks, thickness=2)
        # Top View (XZ)
        landmarks_xz = landmarks_world[:, [0, 2]]
        landmarks_xz[:, 1] = -landmarks_xz[:, 1]
        landmarks_xz = (landmarks_xz * SCALE + np.array([300, 100])).astype(np.int32)
        draw_skeleton_lines(display_3d, landmarks_xz, keep_landmarks, thickness=2)
        # Left View (YZ)
        landmarks_yz = landmarks_world[:, [2, 1]]
        landmarks_yz[:, 0] = -landmarks_yz[:, 0]
        landmarks_yz = (landmarks_yz * SCALE + np.array([100, 300])).astype(np.int32)
        draw_skeleton_lines(display_3d, landmarks_yz, keep_landmarks, thickness=2)
        # Right View (ZY)
        landmarks_zy = landmarks_world[:, [2, 1]]
        landmarks_zy = (landmarks_zy * SCALE + np.array([300, 300])).astype(np.int32)
        draw_skeleton_lines(display_3d, landmarks_zy, keep_landmarks, thickness=2)

    # Copy input image for drawing
    display_screen = image.copy()
    # Create blank 3D display canvas
    display_3d = None
    if draw_3d:
        display_3d = np.zeros((DISPLAY_SIZE, DISPLAY_SIZE, 3), np.uint8)
        # Draw axes and labels for 3D visualization
        cv2.line(display_3d, (DISPLAY_CENTER, 0), (DISPLAY_CENTER, DISPLAY_SIZE), COLOR_WHITE, 2)
        cv2.line(display_3d, (0, DISPLAY_CENTER), (DISPLAY_SIZE, DISPLAY_CENTER), COLOR_WHITE, 2)
        cv2.putText(display_3d, "Main View", (0, 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_RED)
        cv2.putText(display_3d, "Top View", (DISPLAY_CENTER, 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_RED)
        cv2.putText(display_3d, "Left View", (0, DISPLAY_CENTER + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_RED)
        cv2.putText(display_3d, "Right View", (DISPLAY_CENTER, DISPLAY_CENTER + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_RED)

    # Only draw 3D pose for the first detected pose (for clarity)
    drew_3d = False

    for pose_result in poses:
        # Unpack pose result
        bbox, landmarks_screen, landmarks_world, mask, _heatmap, conf = pose_result

        # Optional: draw green edges from mask onto the display image
        if draw_mask_edges:
            display_screen = draw_edges_on_mask(mask, display_screen)

        # Remove last 6 landmarks (as in original code)
        landmarks_screen = landmarks_screen[:-6, :]
        landmarks_world = landmarks_world[:-6, :]
        keep_landmarks = landmarks_screen[:, 4] > 0.8

        # Draw 2D pose (bounding box, skeleton, keypoints)
        draw_pose_2d(display_screen, bbox, conf, landmarks_screen, keep_landmarks)

        # Draw 3D pose projections for the first pose only
        if draw_3d and not drew_3d and display_3d is not None:
            drew_3d = True
            draw_pose_3d(display_3d, landmarks_world, keep_landmarks)

    return display_screen, display_3d

def setup_opencv() -> None:
    """Small performance tweaks for OpenCV on low-power devices."""
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(2)
    except Exception:
        pass


def display_interval_from_config(configs: dict) -> float:
    display_fps = float(configs.get("displayfps", 0) or 0)
    capture_fps = float(configs.get("fps", 0) or 0)

    if display_fps <= 0:
        return 0.0  # no throttling
    if capture_fps > 0 and display_fps >= 0.8 * capture_fps:
        return 0.0  # close to capture fps so no throttling
    return 1.0 / display_fps  # throttled display


def main() -> None:
    setup_opencv()

    # Setting up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("PiCamera2 Capture")

    # Silence Picamera2
    os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:3")  # 3=ERROR, 2=WARNING, 1=INFO, 0=DEBUG

    camera_index = 0

    configs = {
        ################################################################
        # Picamera2 capture configuration
        #
        # List camera properties with:
        #     examples/list_Picamera2Properties.py
        ################################################################
        # Capture mode:
        #   'main' -> full-FOV processed stream (BGR/YUV), scaled to 'camera_res' (libcamera scales)
        #   'raw'  -> high-FPS raw sensor window (exact sensor mode only), cropped FOV
        'mode'            : 'main',
        'camera_res'      : (640, 480),     # requested main stream size (w, h)
        'exposure'        : 0,              # microseconds, 0/-1 for auto
        'fps'             : 60,             # requested capture frame rate
        'autoexposure'    : -1,             # -1 leave unchanged, 0 AE off, 1 AE on
        #'aemeteringmode'  : -1,             # int or 'center'|'spot'|'matrix' 
        'autowb'          : -1,             # -1 leave unchanged, 0 AWB off, 1 AWB on
        #'awbmode'         : -1,             # int or friendly string 'auto'|'incandescent'|'fluorescent'|'warm-fluorescent'|'daylight'|'cloudy-daylight'|'twilight'|'shade'
        # Main stream formats: BGR3 (BGR888), RGB3 (RGB888), YU12 (YUV420), YUY2 (YUYV)
        # Raw stream formats:  SRGGB8, SRGGB10_CSI2P, (see properties script)
        'format'          : 'BGR888',
        "stream_policy"   : "default",      # 'default', 'maximize_fps_no_crop', 'maximize_fps_with_crop', 'maximize_fov'
        'low_latency'     : False,          # low_latency=True prefers size-1 buffer (latest frame)
        'buffersize'      : 4,              # capture queue size override (wrapper-level)
        'buffer_overwrite': True,           # overwrite old frames if buffer full
        'output_res'      : (-1, -1),       # (-1,-1): output == input; else libcamera scales main
        'flip'            : 0,              # 0=norotation 
        'displayfps'      : 30              # frame rate for display server
    }

    display_interval = display_interval_from_config(configs)

    dps_measure_interval = 5.0

    # Display Window Setup

    window_name = "Camera"
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 20)
    textLocation1 = (10, 40)
    textLocation2 = (10, 60)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 1

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Camera Setup

    camera = piCamera2Capture(configs, camera_num=camera_index)
    if not camera.cam_open:
        raise RuntimeError("PiCamera2 camera failed to open")

    # Logging Setup

    logger.log(logging.INFO, "Getting Images")
    logger.log(
        logging.INFO,
        "Config: mode=%s format=%s camera_res=%s output_res=%s",
        configs.get("mode"),
        configs.get("format"),
        configs.get("camera_res"),
        configs.get("output_res"),
    )
    camera.log_stream_options() # Optional show suggested main options or raw sensor modes
    camera.start()
    camera.log_camera_config_and_controls()

    # Load Models
    person_detector, pose_estimator = load_models(logger)

    # Display Window Setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    textLocation0 = (10, 20)
    textLocation1 = (10, 40)
    textLocation2 = (10, 60)
    textLocation3 = (10, 80)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 1

    window_name_image = "MediaPipe Pose Detection Demo"
    window_name_3d = "3D Pose Demo"

    cv2.namedWindow(window_name_image, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(window_name_3d, cv2.WINDOW_AUTOSIZE)

    inference_fps = 0.0

    # Initialize variables for main loop

    last_display = time.perf_counter()
    last_dps_time = last_display
    measured_dps = 0.0
    num_frames_displayed = 0
    logged_camera_controls = False

    stop = False
    loop_interval = 1./(2.*configs.get("fps", 30))

    try:
        while not stop:
            current_time = time.perf_counter()

            # Pull latest available frame.
            if camera.buffer.avail > 0:
                # Use copy=False to avoid extra memcpy; we copy only when displaying.
                frame, _frame_time = camera.buffer.pull(copy=False)
            else:
                frame = None

            # Analysis
            frame_proc = None
            frame_3d = None
            if frame is not None:
                frame_proc = frame.copy()
                infer_start = time.perf_counter()
                persons = person_detector.infer(frame_proc)
                poses = []
                for person in persons:
                    pose = pose_estimator.infer(frame_proc, person)
                    if pose is not None:
                        poses.append(pose)
                infer_elapsed = time.perf_counter() - infer_start
                frame_proc, frame_3d = visualize(
                    frame_proc,
                    poses,
                    draw_3d=False,
                    draw_mask_edges=False,
                    line_thickness=int(1),
                    point_radius=int(1),
                )
            else:
                frame_proc = None
                frame_3d = None

            # Display log
            while not camera.log.empty():
                (level, msg) = camera.log.get_nowait()
                logger.log(level, "{}".format(msg))

            # Display
            delta_display = current_time - last_display
            if (frame is not None) and (delta_display >= display_interval):
                frame_display = frame_proc.copy() if frame_proc is not None else frame.copy()
                if not logged_camera_controls:
                    try:
                        fd = camera.get_control("FrameDuration")
                        fdl = camera.get_control("FrameDurationLimits")
                        sc = camera.get_control("ScalerCrop")
                        logger.log(
                            logging.INFO,
                            "Camera controls: FrameDuration=%s FrameDurationLimits=%s ScalerCrop=%s",
                            fd,
                            fdl,
                            sc,
                        )
                    except Exception:
                        pass
                    logged_camera_controls = True
                cv2.putText(frame_display, "Capture FPS:{:.1f} [Hz]".format(camera.measured_fps),
                    textLocation0, font, fontScale, fontColor, lineType,
                )
                cv2.putText(frame_display, "Display FPS:{:.1f} [Hz]".format(measured_dps),
                    textLocation1, font, fontScale, fontColor, lineType,
                )
                cv2.putText(frame_display, f"Mode:{configs.get('mode')}",
                    textLocation3, font, fontScale, fontColor, lineType,
                )

                if frame_proc is not None:
                    cv2.putText(frame_display, 'Inference:{:<.1f} [ms]'.format(1000.*infer_elapsed),
                        textLocation2, font, fontScale, fontColor, lineType,
                    )

                cv2.imshow(window_name, frame_display)

                # 3D pose visualization
                if frame_3d is not None:
                    cv2.imshow(window_name_3d, frame_3d)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop = True
                # if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                #    stop = True

                last_display = current_time
                num_frames_displayed += 1

            # Update display FPS measurement
            delta_dps = current_time - last_dps_time
            if delta_dps >= dps_measure_interval:
                measured_dps = num_frames_displayed / delta_dps
                num_frames_displayed = 0
                last_dps_time = current_time

            loop_remaining_time = loop_interval - (time.perf_counter() - current_time)
            if loop_remaining_time > 0.:
                time.sleep(loop_remaining_time)

    finally:
        try:
            camera.stop()
            camera.join(timeout=2.0)
            camera.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

""""
(env) uutzinger@urspi:~/pythonBME210/camera $ python3 examples/picamera2_capture_display.py 
INFO:picamera2.picamera2:Initialization successful.
INFO:picamera2.picamera2:Camera now open.
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera configuration has been adjusted!
INFO:picamera2.picamera2:Configuration successful!
INFO:picamera2.picamera2:Camera started
INFO:PiCamera2 Capture:Getting Images
INFO:PiCamera2 Capture:Config: mode=main format=BGR3 camera_res=(640, 480) output_res=(-1, -1)
INFO:PiCamera2 Capture:PiCam2:MAIN sensor selection policy=maximize_fps desired_main=640x480 selected_sensor=(640, 480) bit_depth=10 fps~58.92
INFO:PiCamera2 Capture:PiCam2:ISP configuration successful - hardware will handle format=BGR888, size=(640, 480), transform=False
INFO:PiCamera2 Capture:PiCam2:Controls set {'AeEnable': True, 'AeMeteringMode': 0, 'AwbEnable': True, 'AwbMode': 0}
INFO:PiCamera2 Capture:PiCam2:Open summary stream=main size=(640, 480) fmt=BGR888 req_fps=60 FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920) cpu_resize=False cpu_flip=False cpu_convert=False
INFO:PiCamera2 Capture:PiCam2:Camera opened
INFO:PiCamera2 Capture:PiCam2:Main Stream mode 640x480 format=BGR888. Supported main formats: XBGR8888, XRGB8888, RGB888, BGR888, YUV420, YUYV, MJPEG
INFO:PiCamera2 Capture:PiCam2:Main Stream can scale to arbitrary resolutions; non-native aspect ratios may crop. For raw modes list, run examples/list_Picamera2Properties.py.
INFO:PiCamera2 Capture:PiCam2:Suggested Main Stream options (camera_res/output_res, max_fps, full_fov):
INFO:PiCamera2 Capture:PiCam2:  640x480 -> 640x480 fmt=BGR888 max_fps~58.9 full_fov=False
INFO:PiCamera2 Capture:PiCam2:  1296x972 -> 1296x972 fmt=BGR888 max_fps~46.3 full_fov=False
INFO:PiCamera2 Capture:PiCam2:  1920x1080 -> 1920x1080 fmt=BGR888 max_fps~32.8 full_fov=False
INFO:PiCamera2 Capture:PiCam2:  2592x1944 -> 2592x1944 fmt=BGR888 max_fps~15.6 full_fov=True
INFO:PiCamera2 Capture:PiCam2:=== camera configuration ===
INFO:PiCamera2 Capture:PiCam2:Requested mode=main camera_res=(640, 480) output_res=(-1, -1) format=BGR3 fps=60 stream_policy=maximize_fps low_latency=False flip=0
INFO:PiCamera2 Capture:PiCam2:Requested controls exposure=0 autoexposure=1 aemeteringmode=center autowb=1 awbmode=auto
INFO:PiCamera2 Capture:PiCam2:camera_configuration={'use_case': 'video', 'transform': <libcamera.Transform 'identity'>, 'colour_space': <libcamera.ColorSpace 'SMPTE170M'>, 'buffer_count': 6, 'queue': True, 'main': {'format': 'BGR888', 'size': (640, 480), 'preserve_ar': True, 'stride': 1920, 'framesize': 921600}, 'lores': None, 'raw': {'format': 'GBRG_PISP_COMP1', 'size': (640, 480), 'stride': 640, 'framesize': 307200}, 'controls': {'NoiseReductionMode': <NoiseReductionModeEnum.Fast: 1>, 'FrameDurationLimits': (16667, 16667)}, 'sensor': {'bit_depth': 10, 'output_size': (640, 480)}, 'display': 'main', 'encode': 'main'}
INFO:PiCamera2 Capture:PiCam2:configured controls={'NoiseReductionMode': <NoiseReductionModeEnum.Fast: 1>, 'FrameDurationLimits': (16667, 16667)}
INFO:PiCamera2 Capture:PiCam2:last set_controls={'AeEnable': True, 'AeMeteringMode': 0, 'AwbEnable': True, 'AwbMode': 0}
INFO:PiCamera2 Capture:PiCam2:readback controls unavailable (get_controls missing)
INFO:PiCamera2 Capture:PiCam2:available controls (28): ['AeConstraintMode', 'AeEnable', 'AeExposureMode', 'AeFlickerMode', 'AeFlickerPeriod', 'AeMeteringMode', 'AnalogueGain', 'AnalogueGainMode', 'AwbEnable', 'AwbMode', 'Brightness', 'CnnEnableInputTensor', 'ColourCorrectionMatrix', 'ColourGains', 'ColourTemperature', 'Contrast', 'ExposureTime', 'ExposureTimeMode', 'ExposureValue', 'FrameDurationLimits', 'HdrMode', 'NoiseReductionMode', 'Saturation', 'ScalerCrop', 'Sharpness', 'StatsOutputEnable', 'SyncFrames', 'SyncMode']
INFO:PiCamera2 Capture:PiCam2:camera_properties={'Model': 'ov5647', 'UnitCellSize': (1400, 1400), 'Location': 2, 'Rotation': 0, 'ColorFilterArrangement': 2, 'PixelArraySize': (2592, 1944), 'PixelArrayActiveAreas': [(16, 6, 2592, 1944)], 'ScalerCropMaximum': (16, 0, 2560, 1920), 'SystemDevices': (20752, 20753, 20754, 20755, 20756, 20757, 20758, 20739, 20740, 20741, 20742), 'SensorSensitivity': 1.0}
INFO:PiCamera2 Capture:PiCam2:metadata FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920) AeEnable=None ExposureTime=16836 AwbEnable=None AwbMode=None AeMeteringMode=None AnalogueGain=2.125
INFO:PiCamera2 Capture:Camera controls: FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920)
INFO:picamera2.picamera2:Camera stopped
INFO:picamera2.picamera2:Camera closed successfully.

(env) uutzinger@urspi:~/pythonBME210/camera $ python3 testing/picamera2_direct.py --fps 60
Controls set: {'FrameDurationLimits': (16667, 16667)}
=== camera configuration ===
Requested: mode=main size=(640, 480) format=BGR888 fps=60.0 stream_policy=default low_latency=False flip=0
Requested controls: {'FrameDurationLimits': (16667, 16667)}
last set controls: {'FrameDurationLimits': (16667, 16667)}
camera_configuration: {'use_case': 'video', 'transform': <libcamera.Transform 'identity'>, 'colour_space': <libcamera.ColorSpace 'SMPTE170M'>, 'buffer_count': 6, 'queue': True, 'main': {'format': 'BGR888', 'size': (640, 480), 'preserve_ar': True, 'stride': 1920, 'framesize': 921600}, 'lores': None, 'raw': {'format': 'GBRG_PISP_COMP1', 'size': (640, 480), 'stride': 640, 'framesize': 307200}, 'controls': {'NoiseReductionMode': <NoiseReductionModeEnum.Fast: 1>, 'FrameDurationLimits': (16667, 16667)}, 'sensor': {'bit_depth': 10, 'output_size': (640, 480)}, 'display': 'main', 'encode': 'main'}
configured controls: {'NoiseReductionMode': <NoiseReductionModeEnum.Fast: 1>, 'FrameDurationLimits': (16667, 16667)}
camera_properties: {'Model': 'ov5647', 'UnitCellSize': (1400, 1400), 'Location': 2, 'Rotation': 0, 'ColorFilterArrangement': 2, 'PixelArraySize': (2592, 1944), 'PixelArrayActiveAreas': [(16, 6, 2592, 1944)], 'ScalerCropMaximum': (16, 0, 2560, 1920), 'SystemDevices': (20752, 20753, 20754, 20755, 20756, 20757, 20758, 20739, 20740, 20741, 20742), 'SensorSensitivity': 1.0}
camera_controls defaults (subset): {'AeEnable': (False, True, True), 'AeMeteringMode': (0, 3, 0), 'AwbEnable': (False, True, None), 'AwbMode': (0, 7, 0), 'ExposureTime': (134, 4879289, 20000), 'AeExposureMode': (0, 3, 0), 'AeFlickerPeriod': (100, 1000000, None), 'AnalogueGain': (1.0, 63.9375, 1.0), 'Brightness': (-1.0, 1.0, 0.0), 'ColourGains': (0.0, 32.0, None), 'ColourTemperature': (100, 100000, None), 'Contrast': (0.0, 32.0, 1.0), 'FrameDurationLimits': (16971, 4879899, 33333), 'NoiseReductionMode': (0, 4, 0), 'Saturation': (0.0, 32.0, 1.0), 'ScalerCrop': ((16, 0, 164, 128), (16, 0, 2560, 1920), (16, 0, 2560, 1920)), 'Sharpness': (0.0, 16.0, 1.0)}
metadata: FrameDuration=16971 FrameDurationLimits=None ScalerCrop=(16, 0, 2560, 1920) AeEnable=None ExposureTime=16836 AwbEnable=None AwbMode=None AeMeteringMode=None AnalogueGain=4.5625
FPS (last 2.00s): 67.98 | frames=136
FPS (last 2.00s): 67.50 | frames=271
Total: 330 frames in 4.89s => 67.44 FPS
"""
