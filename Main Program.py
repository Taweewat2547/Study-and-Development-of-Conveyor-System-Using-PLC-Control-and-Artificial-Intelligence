import os
import sys
import time
import math
import struct
import threading
import queue
import sqlite3
import requests
import numpy as np
import cv2
from ultralytics import YOLO
import pymcprotocol
from flask import Flask, jsonify, request, send_file

# ============================================================
# CPU THREADS
# ============================================================
_cpu_count = os.cpu_count() or 1
NUM_THREADS = max(1, _cpu_count - 1) if _cpu_count > 1 else 1

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

print(f"[CPU] Detected cores : {_cpu_count}")
print(f"[CPU] Using threads  : {NUM_THREADS}")

# ============================================================
# PLC CONFIGURATION
# ============================================================
PLC_IP = "192.168.3.250"
PLC_PORT = 5000
PLC_MAX_RETRY = 5

# Send / Write Registers
PLC_DEVICE_SHAPE = "D1000"
PLC_DEVICE_ERROR = "D1004"

# Read Telemetry Registers
PLC_DEVICE_SHAPE_CODE_READ = "D1015"  # 16-bit
PLC_DEVICE_RPM              = "D2010"  # 16-bit Int
PLC_DEVICE_SET_HZ           = "D300"   # 16-bit Int (Set Hz)
PLC_DEVICE_MS               = "D2014"  # 32-bit Float
PLC_DEVICE_CYCLE_SPEED      = "D3002"  # 32-bit Float
PLC_DEVICE_ERROR_32BIT      = "D1020"  # 32-bit Float

# Control Bits
PLC_BIT_M10 = "M10"  # Proximity Entry Trigger
PLC_BIT_M20 = "M20"  # Exit Sensor Trigger
PLC_BIT_M30 = "M30"  # Inductive Sensor
PLC_BIT_M50 = "M50"  # Metal Signal

# Heartbeat Registers
PLC_DEVICE_HEARTBEAT       = "D1100"
PLC_DEVICE_HEARTBEAT_COUNT = "D1102"
HEARTBEAT_INTERVAL = 1.0

# ============================================================
# CONVEYOR & DYNAMIC SCANNING CONFIGURATION (CUSTOM CONFIG)
# ============================================================
SENSOR_TO_CAMERA_DIST_M = 0.35  # M10 อยู่ห่างจากจุดกึ่งกลางกล้อง 35 cm = 0.35 m
DEFAULT_SPEED_MS = 0.2          # ค่าความเร็วสำรองกรณีอ่านจาก PLC ไม่ได้หรือเป็น 0
CAMERA_CENTER_TIME_OFFSET = 0   # ชดเชยเวลาถ่ายภาพ (- ชดเชยให้ถ่ายเร็วขึ้น / + ถ่ายช้าลง)

# ============================================================
# LAPTOP REAL-TIME CONFIGURATION
# ============================================================
LAPTOP_IP = "192.168.3.120"
LAPTOP_PORT = 5000
LAPTOP_URL = f"http://{LAPTOP_IP}:{LAPTOP_PORT}/api/plc_data"

# ============================================================
# PI-SIDE READ API CONFIGURATION (สำหรับ Laptop GUI เรียก pull ข้อมูลรูป)
# ============================================================
API_HOST = "0.0.0.0"
API_PORT = 5001

# ============================================================
# ITEM IMAGE STORAGE (รูปผลลัพธ์ต่อ 1 ชิ้นงาน เก็บถาวรบน Pi)
# ============================================================
ITEM_IMAGE_DIR = "/home/taweewat2547/Desktop/Code/item_images"
os.makedirs(ITEM_IMAGE_DIR, exist_ok=True)

# ============================================================
# SQLITE DB CONFIGURATION & SESSION RESET
# ============================================================
DB_PATH = "/home/taweewat2547/Desktop/Code/plc_data.db"
current_item_number = 1  # ค่าเริ่มต้น

def init_db():
    global current_item_number
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plc_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                item_number INTEGER,
                shape_code INTEGER,
                shape_name TEXT,
                error_val REAL,
                rpm INTEGER,
                set_hz INTEGER,
                speed_ms REAL,
                cycle_speed REAL,
                image_path TEXT
            )
        ''')
        try:
            cursor.execute('ALTER TABLE plc_logs ADD COLUMN image_path TEXT')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('ALTER TABLE plc_logs ADD COLUMN set_hz INTEGER')
        except sqlite3.OperationalError:
            pass
            
        # ดึงค่า item_number ล่าสุดจากฐานข้อมูล เพื่อให้รันต่อไม่ซ้ำกับของเดิม
        cursor.execute('SELECT MAX(item_number) FROM plc_logs')
        row = cursor.fetchone()
        if row and row[0] is not None:
            current_item_number = row[0] + 1
        else:
            current_item_number = 1

        conn.commit()
        conn.close()
        print(f"[SQLITE] Database ready. Next Item Number is set to #{current_item_number}.")
    except Exception as e:
        print(f"[SQLITE] Init Error: {e}")
        current_item_number = 1

init_db()

# ============================================================
# NCNN CONFIGURATION
# ============================================================
MODEL_PATH = "/home/taweewat2547/Desktop/Code/best_ncnn_model"
YOLO_IMAGE_SIZE = 320
YOLO_CONFIDENCE = 0.70

# ============================================================
# CAMERA CONFIGURATION
# ============================================================
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30
DETECTION_BUFFER_SIZE = 15
SHAPE_ANALYSIS_STRIDE = 1
STATUS_PRINT_INTERVAL = 1.0

SAVE_DEBUG_IMAGES = True
DEBUG_IMAGE_DIR = "/home/taweewat2547/Desktop/Code/debug_frames"
if SAVE_DEBUG_IMAGES:
    os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)

# ============================================================
# SCAN ZONE CONFIGURATION
# ============================================================
ZONE_LEFT_RATIO = 0.1
ZONE_RIGHT_RATIO = 0.9
ZONE_MIN_SAMPLES_TO_SEND = 6

# ============================================================
# SHAPE CODE & MAPPING
# ============================================================
COLORS = ["Red", "Blue", "Yellow"]
SHAPES = ["Circle", "Triangle", "Square"]
SHAPE_CODE_MAP = {}
code = 1
for color in COLORS:
    for shape in SHAPES:
        SHAPE_CODE_MAP[f"{color}_{shape}"] = code
        code += 1

SHAPE_CODE_MAP["Metal Detected"] = 10
SHAPE_CODE_MAP["Object Not Found"] = 12

SHAPE_NAME_MAP = {
    1: "Red Circle",     2: "Red Triangle",     3: "Red Square",
    4: "Blue Circle",    5: "Blue Triangle",    6: "Blue Square",
    7: "Yellow Circle",  8: "Yellow Triangle",  9: "Yellow Square",
    10: "Metal Detected",        11: "Error Detected",           12: "Object Not Found"
}

# ============================================================
# ERROR CALCULATION CONFIGURATION
# ============================================================
CONF_ERROR_WEIGHT = 0.4
SHAPE_ERROR_WEIGHT = 0.6
IDEAL_CIRCULARITY = {"Circle": 1.000, "Square": 0.785, "Triangle": 0.605}
SHAPE_ERROR_FALLBACK = 100.0

SHAPE_UPSCALE_FACTOR = 3
SHAPE_MORPH_KERNEL_SIZE = 5
SHAPE_APPROX_EPSILON_RATIO = 0.025

HOLE_DETECTION_THRESHOLD = 0.03
HOLE_ERROR_MULTIPLIER = 1.5
DEFECT_DEPTH_THRESHOLD = 0.08

SHAPE_MATCH_TOLERANCE = 0.05
SHAPE_MATCH_MAX_DIFF = 0.30

ENABLE_OUTLIER_REJECTION = True
IQR_MULTIPLIER = 0.01
OUTLIER_MIN_SAMPLES = 5

# ============================================================
# SYNTHETIC MASTER TEMPLATES
# ============================================================
master_contours = {}

def create_synthetic_templates():
    print("\n========================================")
    print("GENERATING SYNTHETIC TEMPLATES (CODE-ONLY)")
    print("========================================")

    mask_circle = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask_circle, (100, 100), 80, 255, -1)
    cnts_c, _ = cv2.findContours(mask_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    master_contours["Circle"] = cnts_c[0]

    mask_square = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask_square, (20, 20), (180, 180), 255, -1)
    cnts_s, _ = cv2.findContours(mask_square, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    master_contours["Square"] = cnts_s[0]

    mask_tri = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array([[100, 31], [180, 169], [20, 169]], np.int32)
    cv2.fillPoly(mask_tri, [pts], 255)
    cnts_t, _ = cv2.findContours(mask_tri, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    master_contours["Triangle"] = cnts_t[0]

    print("[TEMPLATE] Successfully generated Perfect Circle, Square, and Triangle in RAM.")

create_synthetic_templates()

# ============================================================
# CONNECT PLC & HELPER FUNCTIONS
# ============================================================
def connect_plc():
    print(f"\n[PLC] Connecting to {PLC_IP}:{PLC_PORT} ...")
    try:
        pymc = pymcprotocol.Type3E()
        pymc.connect(PLC_IP, PLC_PORT)
        print("[PLC] Connected")
        return pymc
    except Exception as e:
        print(f"[PLC] Connection Error: {e}")
        return None

def combine_32bit_float(words):
    if not words or len(words) < 2:
        return 0.0
    try:
        low, high = words[0], words[1]
        low_s = low - 65536 if low > 32767 else low
        high_s = high - 65536 if high > 32767 else high
        raw_bytes = struct.pack('hh', low_s, high_s)
        val = struct.unpack('f', raw_bytes)[0]
        return float(round(val, 3))
    except Exception as e:
        return 0.0

def read_plc_bit(device_name):
    global plc
    if plc is None: return 0
    try:
        return plc.batchread_bitunits(headdevice=device_name, readsize=1)[0]
    except Exception:
        return 0

def read_telemetry_from_plc():
    global plc
    if plc is None: return None
    try:
        shape_code = plc.batchread_wordunits(headdevice=PLC_DEVICE_SHAPE_CODE_READ, readsize=1)[0]
        rpm = plc.batchread_wordunits(headdevice=PLC_DEVICE_RPM, readsize=1)[0]
        set_hz = plc.batchread_wordunits(headdevice=PLC_DEVICE_SET_HZ, readsize=1)[0]
        speed_ms = combine_32bit_float(plc.batchread_wordunits(headdevice=PLC_DEVICE_MS, readsize=2))
        cycle_speed = combine_32bit_float(plc.batchread_wordunits(headdevice=PLC_DEVICE_CYCLE_SPEED, readsize=2))
        error_val = combine_32bit_float(plc.batchread_wordunits(headdevice=PLC_DEVICE_ERROR_32BIT, readsize=2))
        shape_name = SHAPE_NAME_MAP.get(shape_code, f"Unknown ({shape_code})")
        return {
            "shape_code": shape_code, "shape_name": shape_name, "rpm": rpm, "set_hz": set_hz,
            "speed_ms": speed_ms, "cycle_speed": cycle_speed, "error_val": error_val
        }
    except Exception as e:
        print(f"[PLC TELEMETRY READ ERROR] {e}")
        return None

def get_conveyor_speed():
    telemetry = read_telemetry_from_plc()
    if telemetry and telemetry.get('speed_ms', 0) > 0:
        return telemetry['speed_ms']
    return DEFAULT_SPEED_MS

def calculate_time_to_center(speed_ms):
    if speed_ms <= 0:
        speed_ms = DEFAULT_SPEED_MS
    time_sec = (SENSOR_TO_CAMERA_DIST_M / speed_ms) + CAMERA_CENTER_TIME_OFFSET
    return max(0.1, time_sec)

def calculate_scan_timeout(speed_ms):
    time_to_center = calculate_time_to_center(speed_ms)
    return max(1.0, time_to_center * 2)

def capture_fresh_frame(cap, flush_count=2):
    frame = None
    for _ in range(flush_count):
        ret, frame = cap.read()
    return frame

def send_realtime_to_laptop(telemetry, item_number, image_path=""):
    if telemetry is None: return
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "item_number": item_number,
        "shape_code": telemetry["shape_code"],
        "shape_name": telemetry["shape_name"],
        "error_val": telemetry["error_val"],
        "rpm": telemetry["rpm"],
        "set_hz": telemetry.get("set_hz", 0),
        "speed_ms": telemetry["speed_ms"],
        "cycle_speed": telemetry["cycle_speed"],
        "image_path": image_path
    }
    def _send_thread(data):
        try:
            resp = requests.post(LAPTOP_URL, json=data, timeout=1.5)
            print(f"[HTTP PUSH SUCCESS] Item #{data['item_number']} ({data['shape_name']}) -> Laptop GUI")
        except Exception as e:
            print(f"[HTTP PUSH ERROR] Cannot send to Laptop: {e}")
    threading.Thread(target=_send_thread, args=(payload,), daemon=True).start()

def save_item_image(item_number, frame, xyxy=None, label_text=None):
    try:
        vis = frame.copy()
        if xyxy is not None:
            h, w = vis.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if label_text:
                ty = max(20, y1 - 8)
                cv2.putText(vis, label_text, (x1 + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, label_text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"item_{item_number:06d}_{timestamp}.jpg"
        filepath = os.path.join(ITEM_IMAGE_DIR, filename)
        cv2.imwrite(filepath, vis)
        return filename
    except Exception as e:
        print(f"[ITEM IMAGE] Save Error: {e}")
        return ""

def write_sqlite_log(telemetry, result_frame=None, result_xyxy=None):
    global current_item_number
    
    if telemetry is None:
        telemetry = {
            "shape_code": 12,
            "shape_name": "Object Not Found",
            "error_val": 0.0,
            "rpm": 0,
            "set_hz": 0,
            "speed_ms": get_conveyor_speed(),
            "cycle_speed": 0.0
        }

    try:
        image_filename = ""
        if result_frame is not None:
            label = f"{telemetry['shape_name']} Err:{telemetry['error_val']:.1f}%"
            image_filename = save_item_image(current_item_number, result_frame, result_xyxy, label)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO plc_logs (item_number, shape_code, shape_name, error_val, rpm, set_hz, speed_ms, cycle_speed, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_item_number,
            telemetry["shape_code"],
            telemetry["shape_name"],
            telemetry["error_val"],
            telemetry["rpm"],
            telemetry.get("set_hz", 0),
            telemetry["speed_ms"],
            telemetry["cycle_speed"],
            image_filename
        ))
        conn.commit()
        conn.close()
        print(f"[SQLITE LOGGED] Item #{current_item_number} | Code {telemetry['shape_code']} ({telemetry['shape_name']}) | Img: {image_filename or '-'}")
        
        send_realtime_to_laptop(telemetry, current_item_number, image_filename)
        current_item_number += 1
    except Exception as e:
        print(f"[SQLITE LOG ERROR] {e}")

heartbeat_toggle = 0
heartbeat_counter = 0

def send_heartbeat():
    global plc, heartbeat_toggle, heartbeat_counter
    try:
        heartbeat_toggle = 1 - heartbeat_toggle
        heartbeat_counter = (heartbeat_counter + 1) % 10000
        plc.batchwrite_wordunits(headdevice=PLC_DEVICE_HEARTBEAT, values=[heartbeat_toggle])
        plc.batchwrite_wordunits(headdevice=PLC_DEVICE_HEARTBEAT_COUNT, values=[heartbeat_counter])
        return True
    except Exception as e:
        print(f"[HEARTBEAT] Write Error: {e}")
        try:
            if plc is not None: plc.close()
        except Exception: pass
        plc = connect_plc()
        return False

# ============================================================
# SCAN ZONE HELPER
# ============================================================
def get_zone(center_x, frame_width):
    left_boundary = frame_width * ZONE_LEFT_RATIO
    right_boundary = frame_width * ZONE_RIGHT_RATIO

    if center_x < left_boundary:
        return "LEFT"
    elif center_x < right_boundary:
        return "MIDDLE"
    else:
        return "RIGHT"

def draw_zone_lines(vis, frame_width, frame_height):
    left_x = int(frame_width * ZONE_LEFT_RATIO)
    right_x = int(frame_width * ZONE_RIGHT_RATIO)
    cv2.line(vis, (left_x, 0), (left_x, frame_height), (255, 200, 0), 1)
    cv2.line(vis, (right_x, 0), (right_x, frame_height), (255, 200, 0), 1)

# ============================================================
# SHAPE GEOMETRY ANALYSIS
# ============================================================
def get_shape_type(class_name):
    parts = class_name.split("_")
    if len(parts) >= 2:
        return parts[-1]
    return class_name

def analyze_shape_geometry(frame, xyxy, shape_type):
    try:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return SHAPE_ERROR_FALLBACK

        crop = frame[y1:y2, x1:x2]

        if SHAPE_UPSCALE_FACTOR > 1:
            crop = cv2.resize(crop, None, fx=SHAPE_UPSCALE_FACTOR, fy=SHAPE_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SHAPE_MORPH_KERNEL_SIZE, SHAPE_MORPH_KERNEL_SIZE))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, _morph_kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return SHAPE_ERROR_FALLBACK

        c = max(contours, key=cv2.contourArea)
        _raw_perimeter = cv2.arcLength(c, True)
        _epsilon = SHAPE_APPROX_EPSILON_RATIO * _raw_perimeter
        c_approx = cv2.approxPolyDP(c, _epsilon, True)

        area = cv2.contourArea(c_approx)
        perimeter = cv2.arcLength(c_approx, True)

        if area <= 0 or perimeter <= 0:
            return SHAPE_ERROR_FALLBACK

        match_error = 0.0
        if shape_type in master_contours:
            master_c = master_contours[shape_type]
            diff = cv2.matchShapes(master_c, c, cv2.CONTOURS_MATCH_I1, 0.0)

            if diff > SHAPE_MATCH_TOLERANCE:
                match_error = ((diff - SHAPE_MATCH_TOLERANCE) / (SHAPE_MATCH_MAX_DIFF - SHAPE_MATCH_TOLERANCE)) * 100.0
                match_error = max(0.0, min(100.0, match_error))

        defect_error = 0.0
        hull_indices = cv2.convexHull(c, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3 and len(c) > 3:
            try:
                defects = cv2.convexityDefects(c, hull_indices)
                if defects is not None:
                    max_defect_depth = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        if depth > max_defect_depth:
                            max_defect_depth = depth

                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(c)
                    defect_ratio = max_defect_depth / min(w_rect, h_rect)

                    if defect_ratio > DEFECT_DEPTH_THRESHOLD:
                        defect_error = (defect_ratio * 100.0) * 4.0
            except Exception:
                pass

        hole_error = 0.0
        solid_mask = np.zeros_like(thresh)
        cv2.drawContours(solid_mask, [c], -1, 255, -1)

        expected_solid_pixels = cv2.countNonZero(solid_mask)
        actual_object_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, solid_mask))

        if expected_solid_pixels > 0:
            missing_pixels = expected_solid_pixels - actual_object_pixels
            hole_ratio = missing_pixels / expected_solid_pixels
            if hole_ratio > HOLE_DETECTION_THRESHOLD:
                hole_error = (hole_ratio * 100.0) * HOLE_ERROR_MULTIPLIER

        num_vertices = len(c_approx)
        vertex_error = 0.0
        aspect_error = 0.0
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(c_approx)
        aspect_ratio = float(w_rect) / h_rect if h_rect > 0 else 0.0

        if shape_type == "Triangle":
            if num_vertices < 3 or num_vertices > 7: vertex_error = 40.0
        elif shape_type == "Square":
            if num_vertices < 4 or num_vertices > 8: vertex_error = 40.0
            if abs(1.0 - aspect_ratio) > 0.15: aspect_error = abs(1.0 - aspect_ratio) * 100.0
        elif shape_type == "Circle":
            if num_vertices < 6: vertex_error = 40.0
            if abs(1.0 - aspect_ratio) > 0.15: aspect_error = abs(1.0 - aspect_ratio) * 100.0

        hull = cv2.convexHull(c_approx)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0: return SHAPE_ERROR_FALLBACK

        solidity = area / hull_area
        solidity_error = max(0.0, min(100.0, (1.0 - solidity) * 100.0))

        circularity = (4.0 * math.pi * area / (perimeter ** 2))
        ideal = IDEAL_CIRCULARITY.get(shape_type, 1.0)
        circularity_error = max(0.0, min(100.0, (abs(circularity - ideal) / ideal) * 100.0))

        shape_error = (solidity_error + circularity_error) / 2.0
        shape_error += hole_error + vertex_error + aspect_error + match_error + defect_error
        shape_error = max(0.0, min(100.0, shape_error))

        return shape_error

    except Exception as e:
        print(f"[SHAPE ANALYSIS] Error: {e}")
        return SHAPE_ERROR_FALLBACK

# ============================================================
# DEBUG IMAGE SAVING
# ============================================================
_debug_save_queue = None
_debug_save_thread = None

def _save_debug_frame_impl(item):
    try:
        frame, xyxy, name, frame_idx, conf, shape_error, zone = item
        vis = frame
        h, w = vis.shape[:2]
        draw_zone_lines(vis, w, h)

        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text_lines = [
            f"{name} ({frame_idx}/{DETECTION_BUFFER_SIZE})",
            f"Zone: {zone}",
            f"Conf: {conf:.3f}",
            f"Shape Error: {shape_error:.1f}%"
        ]

        line_height = 20
        text_block_height = line_height * len(text_lines)
        y_start = y1 - text_block_height - 5 if y1 - text_block_height - 5 >= 0 else y2 + 5

        for i, line in enumerate(text_lines):
            y_pos = y_start + (i + 1) * line_height
            cv2.putText(vis, line, (x1 + 1, y_pos + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, line, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        filename = os.path.join(DEBUG_IMAGE_DIR, f"frame_{frame_idx:02d}.jpg")
        cv2.imwrite(filename, vis)

    except Exception as e:
        print(f"[DEBUG IMAGE] Error: {e}")

def _debug_save_worker():
    while True:
        item = _debug_save_queue.get()
        if item is None:
            _debug_save_queue.task_done()
            break
        _save_debug_frame_impl(item)
        _debug_save_queue.task_done()

def save_debug_frame(frame, xyxy, name, frame_idx, conf, shape_error, zone):
    if _debug_save_queue is None: return
    try:
        frame_copy = frame.copy()
        _debug_save_queue.put_nowait((frame_copy, xyxy, name, frame_idx, conf, shape_error, zone))
    except queue.Full:
        pass

if SAVE_DEBUG_IMAGES:
    _debug_save_queue = queue.Queue(maxsize=DETECTION_BUFFER_SIZE * 2)
    _debug_save_thread = threading.Thread(target=_debug_save_worker, daemon=True)
    _debug_save_thread.start()

# ============================================================
# OUTLIER REJECTION (IQR)
# ============================================================
def reject_outliers_iqr(values, multiplier=IQR_MULTIPLIER):
    n = len(values)
    if n < 4: return values, 0
    sorted_vals = sorted(values)

    def percentile(data, pct):
        idx = (len(data) - 1) * pct
        lower, upper = int(math.floor(idx)), int(math.ceil(idx))
        if lower == upper: return data[lower]
        return data[lower] + (data[upper] - data[lower]) * (idx - lower)

    q1, q3 = percentile(sorted_vals, 0.25), percentile(sorted_vals, 0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - multiplier * iqr, q3 + multiplier * iqr

    filtered = [v for v in values if lower_bound <= v <= upper_bound]
    return filtered, n - len(filtered)

# ============================================================
# WRITE + VERIFY PLC
# ============================================================
def send_to_plc(shape_code, avg_error_percent, name, avg_conf):
    global plc
    error_value = int(round(avg_error_percent * 10))

    print("\n========================================")
    print(f"[PLC SEND] Class: {name} | Code: {shape_code} | Err: {avg_error_percent:.1f}%")
    print("========================================")

    for attempt in range(PLC_MAX_RETRY + 1):
        try:
            plc.batchwrite_wordunits(headdevice=PLC_DEVICE_SHAPE, values=[shape_code])
            plc.batchwrite_wordunits(headdevice=PLC_DEVICE_ERROR, values=[error_value])

            read_shape = plc.batchread_wordunits(headdevice=PLC_DEVICE_SHAPE, readsize=1)
            read_error = plc.batchread_wordunits(headdevice=PLC_DEVICE_ERROR, readsize=1)

            if read_shape[0] == shape_code and read_error[0] == error_value:
                print("[PLC] WRITE + VERIFY SUCCESS")
                return True
            else:
                print("[PLC] VERIFY FAILED. Retrying...")

        except Exception as e:
            print(f"[PLC] WRITE ERROR: {e}")
            if attempt < PLC_MAX_RETRY:
                print("[PLC] Reconnecting...")
                if plc is not None:
                    try: plc.close()
                    except: pass
                plc = connect_plc()
                if plc is None: return False
            else:
                print("[PLC] Maximum retry reached")
    return False

# ============================================================
# CALCULATE RESULT IN RAM
# ============================================================
def calculate_result(name, confs, shape_errors, reason="BUFFER_FULL"):
    avg_conf = sum(confs) / len(confs)

    print()
    print("----------------------------------------")
    print(f"[DEBUG] RAW BUFFER DUMP ({name}) reason={reason}")
    print("----------------------------------------")
    print(f"Conf Samples ({len(confs)}):")
    print("   + " + ", ".join(f"{v:.3f}" for v in confs))
    print(f"Shape Error Samples ({len(shape_errors)}):")
    print("   + " + ", ".join(f"{v:.1f}" for v in shape_errors))
    print("----------------------------------------")

    if ENABLE_OUTLIER_REJECTION:
        filtered_errors, removed_count = reject_outliers_iqr(shape_errors)

        if len(filtered_errors) >= OUTLIER_MIN_SAMPLES:
            shape_errors_for_avg = filtered_errors

            if removed_count > 0:
                _removed_values = [v for v in shape_errors if v not in filtered_errors]
                print(f"[OUTLIER] Removed {removed_count} outlier sample(s) from {len(shape_errors)} readings")
                print("[OUTLIER] Removed values: " + ", ".join(f"{v:.1f}" for v in _removed_values))
        else:
            shape_errors_for_avg = shape_errors
    else:
        shape_errors_for_avg = shape_errors

    avg_shape_error = sum(shape_errors_for_avg) / len(shape_errors_for_avg)
    conf_error_percent = (1.0 - avg_conf) * 100.0

    avg_error_percent = (conf_error_percent * CONF_ERROR_WEIGHT + avg_shape_error * SHAPE_ERROR_WEIGHT)
    avg_error_percent = max(0.0, min(100.0, avg_error_percent))

    print(
        f"[ERROR CALC] Conf Error: {conf_error_percent:.1f}% | "
        f"Shape Error: {avg_shape_error:.1f}% (n={len(shape_errors_for_avg)}) | "
        f"Combined: {avg_error_percent:.1f}%"
    )

    shape_code = SHAPE_CODE_MAP.get(name, 12)
    return shape_code, avg_error_percent, name, avg_conf

# ============================================================
# LOAD NCNN MODEL
# ============================================================
print("\n========================================")
print("LOADING NCNN MODEL")
print("========================================")
if not os.path.isdir(MODEL_PATH):
    print(f"[NCNN] ERROR: Model folder not found! Expected {MODEL_PATH}")
    sys.exit(1)

try:
    model = YOLO(MODEL_PATH)
    print(f"[NCNN] Model Loaded. Classes: {model.names}")
except Exception as e:
    print(f"[NCNN] Model Error: {e}")
    sys.exit(1)

print("\n[NCNN] Warming up model...")
_ = model(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8), imgsz=YOLO_IMAGE_SIZE, conf=YOLO_CONFIDENCE, verbose=False)

# ============================================================
# PI-SIDE READ API
# ============================================================
api_app = Flask(__name__)

@api_app.route('/api/health')
def api_health():
    return jsonify({"status": "ok", "current_item_number": current_item_number})

@api_app.route('/api/history')
def api_history():
    limit = request.args.get('limit', default=200, type=int)
    offset = request.args.get('offset', default=0, type=int)
    limit = max(1, min(limit, 2000))
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, item_number, shape_code, shape_name,
                   error_val, rpm, set_hz, speed_ms, cycle_speed, image_path
            FROM plc_logs
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        rows = [dict(r) for r in cursor.fetchall()]
        cursor.execute('SELECT COUNT(*) FROM plc_logs')
        total = cursor.fetchone()[0]
        conn.close()
        return jsonify({"status": "success", "count": len(rows), "total": total, "data": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_app.route('/api/image/<int:item_number>')
def api_image(item_number):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT image_path FROM plc_logs WHERE item_number = ? ORDER BY id DESC LIMIT 1',
            (item_number,)
        )
        row = cursor.fetchone()
        conn.close()
        if not row or not row[0]:
            return jsonify({"status": "error", "message": "No image for this item"}), 404
        filepath = os.path.join(ITEM_IMAGE_DIR, row[0])
        if not os.path.exists(filepath):
            return jsonify({"status": "error", "message": "Image file missing on disk"}), 404
        return send_file(filepath, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def run_api_server():
    try:
        api_app.run(host=API_HOST, port=API_PORT, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"[API SERVER] Error: {e}")

_api_thread = threading.Thread(target=run_api_server, daemon=True)
_api_thread.start()
print(f"[API SERVER] Read-only history/image API listening on port {API_PORT}")

# ============================================================
# INITIALIZE SYSTEM
# ============================================================
plc = connect_plc()
if plc is None:
    print("[ERROR] Cannot connect to PLC. Program stopped."); sys.exit(1)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[CAMERA] Camera Error"); sys.exit(1)

print("\n========================================")
print("STARTING MAIN LOOP (Press Ctrl+C to stop)")
print("========================================")

prev_time = time.time()
last_status_print = 0.0
last_heartbeat_time = 0.0

prev_m10_state = 0
prev_m20_state = 0

is_scanning = False
scan_start_time = 0.0
dynamic_scan_timeout = 3.0

time_to_center_target = 0.0
center_photo_captured = False

pending_result = None  
has_scanned = False
already_sent = False  # Flag ป้องกันส่งข้อมูลซ้ำต่อ 1 ชิ้นงาน

tracked_name = None
tracked_confs = []
tracked_shape_errors = []
tracked_frame_count = 0
current_zone_text = ""

last_tracked_frame = None
last_tracked_xyxy = None
pending_result_frame = None
pending_result_xyxy = None

# ============================================================
# MAIN LOOP
# ============================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret: print("[CAMERA] Read Error"); break

        frame_h, frame_w = frame.shape[:2]

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time - prev_time > 0 else 0.0
        prev_time = current_time

        # ---- HEARTBEAT ----
        if current_time - last_heartbeat_time >= HEARTBEAT_INTERVAL:
            send_heartbeat()
            last_heartbeat_time = current_time

        # ---- READ CONTROL BITS ----
        current_m10 = read_plc_bit(PLC_BIT_M10)
        current_m20 = read_plc_bit(PLC_BIT_M20)
        current_m50 = read_plc_bit(PLC_BIT_M50)

        # ---- RISING EDGE M10 (เข้า Scan Zone) ----
        if prev_m10_state == 0 and current_m10 == 1:
            print("\n[TRIGGER M10] Rising Edge Detected - START SCANNING ZONE")
            
            current_speed = get_conveyor_speed()
            time_to_center_target = calculate_time_to_center(current_speed)
            dynamic_scan_timeout = calculate_scan_timeout(current_speed)
            
            print(f"[TIMING CONFIG] Speed: {current_speed:.3f} m/s | Time-to-Center: {time_to_center_target:.2f}s | Scan Timeout: {dynamic_scan_timeout:.2f}s")

            is_scanning = True
            scan_start_time = current_time
            center_photo_captured = False
            pending_result = None
            has_scanned = False
            already_sent = False  # รีเซ็ต Flag เป็น False สำหรับชิ้นงานใหม่
            
            tracked_name = None
            tracked_confs = []
            tracked_shape_errors = []
            tracked_frame_count = 0
            last_tracked_frame = None
            last_tracked_xyxy = None
            pending_result_frame = None
            pending_result_xyxy = None

        prev_m10_state = current_m10

        # ---- SCANNING LOGIC ----
        if is_scanning and not has_scanned:

            elapsed_scan_time = current_time - scan_start_time

            # 1. ถ่ายภาพบันทึก ณ จุดกึ่งกลางกล้อง
            if not center_photo_captured and elapsed_scan_time >= time_to_center_target:
                center_frame = capture_fresh_frame(cap, flush_count=2)
                if center_frame is not None:
                    last_tracked_frame = center_frame.copy()
                    if current_m50 != 1:
                        center_results = model(center_frame, imgsz=YOLO_IMAGE_SIZE, conf=YOLO_CONFIDENCE, verbose=False)
                        if len(center_results[0].boxes) > 0:
                            last_tracked_xyxy = center_results[0].boxes[0].xyxy[0]
                center_photo_captured = True
                print(f"[CENTER PHOTO] Captured fresh frame at center position ({elapsed_scan_time:.2f}s)")

            # 2. กรณีวัตถุเป็นโลหะ (M50 Signal Active)
            if current_m50 == 1:
                print(f"[LOGIC] M50 Active - Metal Detected")
                if center_photo_captured:
                    pending_result = (10, 0.0, "Metal Detected", 1.0)
                    pending_result_frame = last_tracked_frame
                    pending_result_xyxy = None
                    has_scanned = True
                    is_scanning = False
                continue

            # 3. กรณีวัตถุทั่วไป -> ใช้ YOLO ตรวจจับรูปทรง
            results = model(frame, imgsz=YOLO_IMAGE_SIZE, conf=YOLO_CONFIDENCE, verbose=False)
            boxes = results[0].boxes

            best_box = None
            best_conf = -1.0

            for box in boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = (model.names[int(box.cls[0])], conf, box.xyxy[0])

            if best_box is None:
                if (
                    tracked_name is not None
                    and len(tracked_confs) >= ZONE_MIN_SAMPLES_TO_SEND
                ):
                    pending_result = calculate_result(
                        tracked_name, tracked_confs, tracked_shape_errors, reason="LOST_TRACK_GRACE"
                    )
                    pending_result_frame = last_tracked_frame if last_tracked_frame is not None else frame.copy()
                    pending_result_xyxy = last_tracked_xyxy
                    has_scanned = True
                    is_scanning = False

                tracked_name = None
                tracked_confs = []
                tracked_shape_errors = []
                tracked_frame_count = 0
                current_zone_text = ""

            else:
                name, conf, xyxy = best_box

                cx = (float(xyxy[0]) + float(xyxy[2])) / 2.0
                zone = get_zone(cx, frame_w)
                current_zone_text = zone

                if name == tracked_name:
                    if zone == "MIDDLE":
                        tracked_confs.append(conf)
                        tracked_frame_count += 1

                        if tracked_frame_count % SHAPE_ANALYSIS_STRIDE == 0:
                            shape_type = get_shape_type(name)
                            last_shape_error = analyze_shape_geometry(frame, xyxy, shape_type)
                            tracked_shape_errors.append(last_shape_error)

                            if SAVE_DEBUG_IMAGES:
                                save_debug_frame(frame, xyxy, name, tracked_frame_count, conf, last_shape_error, zone)

                    elif zone == "RIGHT":
                        if len(tracked_confs) >= ZONE_MIN_SAMPLES_TO_SEND:
                            pending_result = calculate_result(
                                tracked_name, tracked_confs, tracked_shape_errors, reason="EARLY_EXIT_RIGHT_ZONE"
                            )
                            pending_result_frame = last_tracked_frame if last_tracked_frame is not None else frame.copy()
                            pending_result_xyxy = last_tracked_xyxy
                            has_scanned = True
                            is_scanning = False

                else:
                    tracked_name = name
                    tracked_confs = []
                    tracked_shape_errors = []
                    tracked_frame_count = 0

                    if zone == "MIDDLE":
                        tracked_confs = [conf]
                        tracked_frame_count = 1

                        shape_type = get_shape_type(name)
                        last_shape_error = analyze_shape_geometry(frame, xyxy, shape_type)
                        tracked_shape_errors = [last_shape_error]

                        if SAVE_DEBUG_IMAGES:
                            save_debug_frame(frame, xyxy, name, tracked_frame_count, conf, last_shape_error, zone)

                if len(tracked_confs) >= DETECTION_BUFFER_SIZE:
                    pending_result = calculate_result(
                        tracked_name, tracked_confs, tracked_shape_errors, reason="BUFFER_FULL"
                    )
                    pending_result_frame = last_tracked_frame if last_tracked_frame is not None else frame.copy()
                    pending_result_xyxy = last_tracked_xyxy
                    has_scanned = True
                    is_scanning = False

            # [LOGIC 1] Timeout Not Found -> บันทึก DB และส่งไป GUI ทันทีโดยไม่ต้องรอ M20
            if not has_scanned and not already_sent and elapsed_scan_time >= dynamic_scan_timeout:
                print(f"[LOGIC] Timeout ({dynamic_scan_timeout:.2f}s) - Object Not Found (Sending immediately)")
                nof_frame = last_tracked_frame if last_tracked_frame is not None else frame.copy()
                
                send_to_plc(12, 0.0, "Object Not Found", 0.0)
                
                time.sleep(0.05)
                telemetry_data = read_telemetry_from_plc()
                if telemetry_data is None or telemetry_data.get("shape_code") == 0:
                    telemetry_data = {
                        "shape_code": 12,
                        "shape_name": "Object Not Found",
                        "rpm": 0,
                        "set_hz": 0,
                        "speed_ms": get_conveyor_speed(),
                        "cycle_speed": 0.0,
                        "error_val": 0.0
                    }
                
                write_sqlite_log(telemetry_data, nof_frame, None)
                
                already_sent = True     # ทำเครื่องหมายว่าส่งไปแล้ว
                pending_result = None   # ล้างค่าทิ้งเพื่อป้องกัน M20 นำไปส่งซ้ำ
                has_scanned = True
                is_scanning = False

        # ============================================================
        # RISING EDGE M20 (ส่งค่า PLC/DB)
        # ============================================================
        if prev_m20_state == 0 and current_m20 == 1:
            print("\n[TRIGGER M20] Rising Edge Detected")
            
            # เช็คว่าถ้าส่งไปแล้วตอน Timeout ให้ข้ามเลย ไม่ส่งซ้ำ
            if already_sent:
                print("[LOGIC] M20 Triggered - Skipped (Data already sent via Timeout)")
            else:
                # [LOGIC 2] หากยังไม่เคยส่ง และไม่มีการ Tracking ใดๆ เกิดขึ้นเลยตั้งแต่ผ่าน M10 มา
                if pending_result is None:
                    print("[LOGIC] M20 Triggered with no tracking - Force Object Not Found")
                    pending_result = (12, 0.0, "Object Not Found", 0.0)
                    pending_result_frame = last_tracked_frame if last_tracked_frame is not None else frame.copy()
                    pending_result_xyxy = None

                shape_code, avg_err, name_label, avg_conf = pending_result
                
                send_to_plc(shape_code, avg_err, name_label, avg_conf)
                
                time.sleep(0.05)
                telemetry_data = read_telemetry_from_plc()
                
                if telemetry_data is None or telemetry_data.get("shape_code") == 0:
                    telemetry_data = {
                        "shape_code": shape_code,
                        "shape_name": SHAPE_NAME_MAP.get(shape_code, name_label),
                        "rpm": 0,
                        "set_hz": 0,
                        "speed_ms": get_conveyor_speed(),
                        "cycle_speed": 0.0,
                        "error_val": avg_err
                    }

                write_sqlite_log(telemetry_data, pending_result_frame, pending_result_xyxy)

                already_sent = True     # ทำเครื่องหมายว่าส่งแล้ว
                has_scanned = True      # ป้องกันไม่ให้ scanning block (section 1-3) ทำงานต่อในเฟรมถัดไป
                is_scanning = False     # ป้องกันไม่ให้ LOGIC 1 (timeout) เข้าเงื่อนไขแล้วส่งซ้ำ
                pending_result = None
                pending_result_frame = None
                pending_result_xyxy = None

        prev_m20_state = current_m20

        # ---- STATUS PRINT ----
        if current_time - last_status_print >= STATUS_PRINT_INTERVAL:
            zone_text = f" | Zone: {current_zone_text}" if current_zone_text else ""
            tracking_text = f" | Track: {tracked_name} ({len(tracked_confs)}/{DETECTION_BUFFER_SIZE})" if tracked_name else ""
            scan_text = f"SCANNING ({current_time - scan_start_time:.1f}s / {dynamic_scan_timeout:.1f}s)" if is_scanning else ("READY (Wait M20)" if pending_result else "IDLE")
            print(f"[STATUS] FPS: {fps:.1f} | State: {scan_text}{zone_text}{tracking_text} | M10: {current_m10} | M20: {current_m20} | M50: {current_m50}")
            last_status_print = current_time

except KeyboardInterrupt:
    print("\n[SYSTEM] Interrupted by user")
except Exception as e:
    print(f"\n[SYSTEM] ERROR: {e}")
finally:
    print("\n========================================")
    print("STOP SYSTEM")
    print("========================================")
    try: cap.release()
    except: pass
    if plc is not None:
        try: plc.close()
        except: pass
    if _debug_save_queue is not None:
        try:
            _debug_save_queue.put(None)
            _debug_save_thread.join(timeout=3.0)
        except: pass
    print("[SYSTEM] STOP")