# Spectacle Eye Detector — Full Version + Manual Adjustment + Reset Options
# - Auto-detect pupils on each frame
# - Click & drag pupil markers to manually correct positions
# - R: reset manual marks to latest automatic detection
# - I: reset manual marks to the initial (very first) automatic detection
# - Thinner pupil markers for clear view
# - Capture (C) with review + Save / Discard / Retake
# - On capture: adjust markers and readings before saving
# - AR focus mark at TOP for user alignment (live only)
# - Only open eyes are editable and shown in capture review
# - Bullet mark is only on open eye(s) in saved image

import cv2
import mediapipe as mp
import math
import os
import time
import numpy as np
import json
import base64
HEADLESS = os.environ.get("HEADLESS", "1") == "1"
from collections import deque

# ---------- Settings ----------
iris_real_mm = 11.7
HISTORY_LEN = 15
HEAD_TILT_LIMIT_DEG = 6.0
ALIGNMENT_IRIS_PX_MIN = 6
ALIGNMENT_IRIS_PX_MAX = 45
CAPTURE_SAVE_DIR = "./captures"
os.makedirs(CAPTURE_SAVE_DIR, exist_ok=True)

# ---------- MediaPipe init ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# ---------- State / smoothing ----------
ipd_history = deque(maxlen=HISTORY_LEN)
left_nose_history = deque(maxlen=HISTORY_LEN)
right_nose_history = deque(maxlen=HISTORY_LEN)
nose_line_history = deque(maxlen=HISTORY_LEN)
scale_history = deque(maxlen=HISTORY_LEN)

# ---------- Manual override & auto storage ----------
manual_left_pupil = None
manual_right_pupil = None
selected_eye = None  # "left" or "right"

auto_left_pupil = None         # latest automatic detected left pupil (subject left / viewer left)
auto_right_pupil = None        # latest automatic detected right pupil (subject right / viewer right)
initial_auto_left_pupil = None # the very first automatic detection (persist)
initial_auto_right_pupil = None

captured_items = []

# ---------- Helpers ----------
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1]) if a and b else 0.0

def angle_between_points_deg(p1, p2):
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    return math.degrees(math.atan2(dy, dx))

def draw_sniper_cross(img, pos, size=20, color=(255,180,0), thickness=2):
    if pos is None:
        return
    x, y = int(pos[0]), int(pos[1])
    cv2.line(img, (x-size, y), (x+size, y), color, thickness)
    cv2.line(img, (x, y-size), (x, y+size), color, thickness)
    cv2.circle(img, (x, y), max(2, size//6), color, -1)

def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    if len(pts) != 6:
        return 0.0
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def draw_instructions(frame, text_lines):
    h, w, _ = frame.shape
    overlay = frame.copy()
    alpha = 0.85
    cv2.rectangle(overlay, (20, 20), (w-20, h-20), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    y = 80
    for line in text_lines:
        cv2.putText(frame, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        y += 34
    cv2.putText(frame, "Press Enter to START    |    S: Skip instructions", (60, y+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

def show_capture_review(img_clean, left_pupil, right_pupil, scale_to_use, avg_nose_line_point, left_open, right_open):
    reviewing = True
    dragging = None
    offset = (0, 0)
    updated_left = left_pupil
    updated_right = right_pupil
    h, w, _ = img_clean.shape

    def mouse_event(event, x, y, flags, param):
        nonlocal dragging, updated_left, updated_right, offset
        if event == cv2.EVENT_LBUTTONDOWN:
            if left_open and updated_left and dist((x, y), updated_left) < 30:
                dragging = 'left'
                offset = (updated_left[0] - x, updated_left[1] - y)
            elif right_open and updated_right and dist((x, y), updated_right) < 30:
                dragging = 'right'
                offset = (updated_right[0] - x, updated_right[1] - y)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            if dragging == 'left' and left_open:
                updated_left = (x + offset[0], y + offset[1])
            elif dragging == 'right' and right_open:
                updated_right = (x + offset[0], y + offset[1])
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = None

    cv2.namedWindow("Capture Review")
    cv2.setMouseCallback("Capture Review", mouse_event)

    while reviewing:
        review_disp = img_clean.copy()
        # Draw markers only for open eyes
        if left_open:
            draw_sniper_cross(review_disp, updated_left, size=8, color=(0,255,0), thickness=1)
        if right_open:
            draw_sniper_cross(review_disp, updated_right, size=8, color=(0,255,0), thickness=1)
        y0 = 40
        # Show readings only for open eyes
        if left_open and right_open:
            pd_px = dist(updated_left, updated_right)
            pd_mm = int(round(pd_px * scale_to_use)) if scale_to_use else 0
            left_nose_mm = int(round(dist(updated_left, avg_nose_line_point) * scale_to_use)) if avg_nose_line_point else 0
            right_nose_mm = int(round(dist(updated_right, avg_nose_line_point) * scale_to_use)) if avg_nose_line_point else 0
            cv2.putText(review_disp, f"PD: {pd_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            y0 += 30
            cv2.putText(review_disp, f"Left→Nose: {left_nose_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            y0 += 30
            cv2.putText(review_disp, f"Right→Nose: {right_nose_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        elif left_open:
            left_nose_mm = int(round(dist(updated_left, avg_nose_line_point) * scale_to_use)) if avg_nose_line_point else 0
            cv2.putText(review_disp, f"Left→Nose: {left_nose_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            y0 += 30
            cv2.putText(review_disp, "Right eye: CLOSED", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        elif right_open:
            right_nose_mm = int(round(dist(updated_right, avg_nose_line_point) * scale_to_use)) if avg_nose_line_point else 0
            cv2.putText(review_disp, f"Right→Nose: {right_nose_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            y0 += 30
            cv2.putText(review_disp, "Left eye: CLOSED", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv2.putText(review_disp, "Both eyes: CLOSED", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        # Instructions
        cv2.putText(review_disp, "Drag green marks to adjust. S: Save  R: Retake  Esc: Cancel", (20, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.imshow("Capture Review", review_disp)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('s'), ord('S')]:
            cv2.destroyWindow("Capture Review")
            return updated_left, updated_right, True  # Save
        elif k in [ord('r'), ord('R')]:
            cv2.destroyWindow("Capture Review")
            return None, None, False  # Retake
        elif k == 27:
            cv2.destroyWindow("Capture Review")
            return None, None, None  # Cancel

# ---------- Mouse callback for manual pupil adjustment ----------
def mouse_callback(event, x, y, flags, param):
    global manual_left_pupil, manual_right_pupil, selected_eye, auto_left_pupil, auto_right_pupil
    # Only act on left button
    if event == cv2.EVENT_LBUTTONDOWN:
        # click near whichever pupil
        if auto_left_pupil and dist((x, y), auto_left_pupil) < 30:
            manual_left_pupil = (x, y)
            selected_eye = "left"
        elif auto_right_pupil and dist((x, y), auto_right_pupil) < 30:
            manual_right_pupil = (x, y)
            selected_eye = "right"
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        if selected_eye == "left":
            manual_left_pupil = (x, y)
        elif selected_eye == "right":
            manual_right_pupil = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selected_eye = None

# ---------- Camera & window ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

if not HEADLESS:
    cv2.namedWindow("Spectacle Eye Detector")
    cv2.setMouseCallback("Spectacle Eye Detector", mouse_callback)

# ---------- Instruction screen ----------
if not HEADLESS:
    show_instructions = True
    while show_instructions:
        ret, frame = cap.read()
        if not ret:
            break
        frame_disp = frame.copy()
        txt = [
            "FINDING EYE CENTRE - PD Measurement",
            "- Keep head level, tilt <= 5-6 degrees",
            "- Position camera at eye level",
            "- Click & drag green pupil markers to adjust manually",
            "- R: Reset to latest auto   |   I: Reset to initial auto",
            "- C: Capture   |   Esc: Exit"
        ]
        draw_instructions(frame_disp, txt)
        cv2.imshow("Spectacle Eye Detector", frame_disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key in [ord('s'), ord('S')]:
            show_instructions = False
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # defaults for this frame
    ipd_mm = None
    ipd_mm_avg = left_nose_avg = right_nose_avg = None
    head_tilt_deg = 0.0
    nose_center = None
    avg_nose_line_point = None
    iris_px = 0.0
    VIEWER_LEFT_OPEN = VIEWER_RIGHT_OPEN = False
    alignment_green = False

    # --- Distance estimation variables ---
    distance_mm = None
    FOCAL_LENGTH_PX = 850  # Adjust for your camera if needed

    # compute automatic detections (if any)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        try:
            # Auto pupil positions from MediaPipe (subject LEFT/RIGHT correspond to viewer LEFT/RIGHT)
            auto_left = (int(face_landmarks.landmark[468].x * w), int(face_landmarks.landmark[468].y * h))
            auto_right = (int(face_landmarks.landmark[473].x * w), int(face_landmarks.landmark[473].y * h))
            nose_center = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))

            # store/update latest automatic positions
            auto_left_pupil = auto_left
            auto_right_pupil = auto_right
            if initial_auto_left_pupil is None:
                initial_auto_left_pupil = auto_left
            if initial_auto_right_pupil is None:
                initial_auto_right_pupil = auto_right

            # Choose final pupil coords: manual override if set, otherwise automatic
            final_left = manual_left_pupil if manual_left_pupil is not None else auto_left_pupil
            final_right = manual_right_pupil if manual_right_pupil is not None else auto_right_pupil

            # Eye closure detection (EAR) - using MediaPipe landmarks (not manual)
            left_eye_idx  = [362, 385, 387, 263, 373, 386]
            right_eye_idx = [33, 160, 158, 133, 153, 159]
            ear_left  = eye_aspect_ratio(face_landmarks.landmark, left_eye_idx, w, h)
            ear_right = eye_aspect_ratio(face_landmarks.landmark, right_eye_idx, w, h)
            VIEWER_LEFT_OPEN  = ear_left  > 0.20
            VIEWER_RIGHT_OPEN = ear_right > 0.20

            # Iris scale (auto)
            left_iris_px = dist((face_landmarks.landmark[469].x * w, face_landmarks.landmark[469].y * h),
                                (face_landmarks.landmark[471].x * w, face_landmarks.landmark[471].y * h))
            right_iris_px = dist((face_landmarks.landmark[474].x * w, face_landmarks.landmark[474].y * h),
                                 (face_landmarks.landmark[476].x * w, face_landmarks.landmark[476].y * h))
            iris_candidates = [v for v in [left_iris_px, right_iris_px] if v and v > 0]
            if iris_candidates:
                iris_px = sum(iris_candidates) / len(iris_candidates)
                current_scale = iris_real_mm / iris_px if iris_px > 0 else 0.1
                if current_scale > 0:
                    scale_history.append(current_scale)
            smoothed_scale = (sum(scale_history) / len(scale_history)) if len(scale_history) else None
            scale_to_use = smoothed_scale if smoothed_scale and smoothed_scale > 0 else (
                iris_real_mm / iris_px if iris_px > 0 else 0.1)

            # --- Estimate distance from camera using iris size ---
            if iris_px > 0:
                distance_mm = int(round((iris_real_mm * FOCAL_LENGTH_PX) / iris_px))
            else:
                distance_mm = None

            # Nose-line: compute horizontal line y using final pupils (manual override affects this)
            if final_left and final_right:
                eye_line_y = (final_left[1] + final_right[1]) / 2.0
                head_tilt_deg = angle_between_points_deg(final_left, final_right)
            else:
                eye_line_y = h // 2
                head_tilt_deg = 0.0

            raw_nose_line_point = (nose_center[0], int(eye_line_y))
            nose_line_history.append(raw_nose_line_point)
            avg_nose_line_point = (int(sum(pt[0] for pt in nose_line_history) / len(nose_line_history)),
                                   int(sum(pt[1] for pt in nose_line_history) / len(nose_line_history)))

            # Distances (use final pupils)
            if VIEWER_LEFT_OPEN and final_left:
                left_to_nose_px = dist(final_left, avg_nose_line_point)
                left_nose_val = left_to_nose_px * scale_to_use
                left_nose_history.append(left_nose_val)
                left_nose_avg = sum(left_nose_history) / len(left_nose_history)
            else:
                left_nose_avg = None

            if VIEWER_RIGHT_OPEN and final_right:
                right_to_nose_px = dist(final_right, avg_nose_line_point)
                right_nose_val = right_to_nose_px * scale_to_use
                right_nose_history.append(right_nose_val)
                right_nose_avg = sum(right_nose_history) / len(right_nose_history)
            else:
                right_nose_avg = None

            # PD logic (final pupils)
            if VIEWER_LEFT_OPEN and VIEWER_RIGHT_OPEN and final_left and final_right:
                ipd_px = dist(final_left, final_right)
                ipd_mm = ipd_px * scale_to_use
                ipd_history.append(ipd_mm)
                ipd_mm_avg = sum(ipd_history) / len(ipd_history)
            elif VIEWER_LEFT_OPEN and not VIEWER_RIGHT_OPEN:
                ipd_mm_avg = left_nose_avg
            elif VIEWER_RIGHT_OPEN and not VIEWER_LEFT_OPEN:
                ipd_mm_avg = right_nose_avg
            else:
                ipd_mm_avg = None

            # Alignment check (uses nose_center + iris + head tilt)
            distance_ok = (iris_px and ALIGNMENT_IRIS_PX_MIN <= iris_px <= ALIGNMENT_IRIS_PX_MAX)
            tilt_ok = abs(head_tilt_deg) <= HEAD_TILT_LIMIT_DEG
            center = (w//2, h//2)
            axis_x, axis_y = int(w*0.23), int(h*0.40)  # elongated oval
            def point_in_ellipse(pt):
                if not pt: return False
                dx = (pt[0] - center[0]) / axis_x
                dy = (pt[1] - center[1]) / axis_y
                return dx*dx + dy*dy <= 1.0
            alignment_green = distance_ok and tilt_ok and point_in_ellipse(nose_center)

        except Exception as e:
            print("Partial detection:", e)
            final_left = manual_left_pupil if manual_left_pupil is not None else auto_left_pupil
            final_right = manual_right_pupil if manual_right_pupil is not None else auto_right_pupil

    else:
        final_left = manual_left_pupil if manual_left_pupil is not None else auto_left_pupil
        final_right = manual_right_pupil if manual_right_pupil is not None else auto_right_pupil

    # ---------- Compose display frame ----------
    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        axis_x, axis_y
    except NameError:
        axis_x, axis_y = int(w*0.23), int(h*0.40)
    center = (w//2, h//2)
    cv2.ellipse(mask, center, (axis_x, axis_y), 0, 0, 360, 255, -1)
    sharp_inside = cv2.bitwise_and(frame, frame, mask=mask)
    mask_inv = cv2.bitwise_not(mask)
    blurred_outside = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
    disp_clean = cv2.add(sharp_inside, blurred_outside)  # <--- CLEAN IMAGE FOR CAPTURE/REVIEW

    disp = disp_clean.copy()

    color_main = (0, 220, 0) if alignment_green else (130, 190, 255)
    cv2.ellipse(disp, center, (axis_x, axis_y), 0, 0, 360, color_main, 4)
    cv2.ellipse(disp, center, (axis_x-6, axis_y-6), 0, 0, 360, (255,255,255), 2)
    if alignment_green:
        overlay = disp.copy()
        cv2.ellipse(overlay, center, (axis_x, axis_y), 0, 0, 360, (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, disp, 0.9, 0, disp)

    # --- AR focus mark at TOP center (live only) ---
    dot_center = (w//2, int(h*0.13))
    cv2.circle(disp, dot_center, 5, (0, 0, 255), -1)  # Smaller red dot
    cv2.circle(disp, dot_center, 10, (255, 255, 255), 1)  # Smaller white ring
    cv2.putText(disp, "Look at the red dot", (dot_center[0]-55, dot_center[1]+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

    # --- Show distance feedback for "one hand distance" at bottom of frame ---
    if iris_px > 0 and distance_mm:
        msg_y = h - 60
        msg_x = 60  # Move to left
        if 550 <= distance_mm <= 750:
            cv2.putText(disp, "Frame is accurately aligned (One hand distance)", (msg_x, msg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
            alignment_green = True
        else:
            cv2.putText(disp, f"Move {'closer' if distance_mm > 750 else 'farther'} to camera", (msg_x, msg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

    fixed_cross = (w // 2, int(h * 0.42))
    overlay_cross = disp.copy()
    draw_sniper_cross(overlay_cross, fixed_cross, size=26, color=(120,120,60), thickness=1)
    cv2.addWeighted(overlay_cross, 0.4, disp, 0.6, 0, disp)

    if avg_nose_line_point:
        cv2.drawMarker(disp, avg_nose_line_point, (80,80,80), cv2.MARKER_TILTED_CROSS, 12, 2)

    pupil_marker_size = 8
    pupil_marker_thickness = 1
    if 'final_left' in locals() and final_left:
        cv2.drawMarker(disp, final_left, (0,255,0), cv2.MARKER_CROSS, pupil_marker_size, pupil_marker_thickness)
    if 'final_right' in locals() and final_right:
        cv2.drawMarker(disp, final_right, (0,255,0), cv2.MARKER_CROSS, pupil_marker_size, pupil_marker_thickness)

    manual_status = []
    if manual_left_pupil: manual_status.append("L")
    if manual_right_pupil: manual_status.append("R")
    if manual_status:
        status_txt = "Manual: " + ",".join(manual_status)
        cv2.putText(disp, status_txt, (w-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    y0 = 30
    cv2.putText(disp, f"PD: {int(round(ipd_mm_avg)) if ipd_mm_avg else '--'} mm", (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    y0 += 30
    if VIEWER_LEFT_OPEN and left_nose_avg is not None:
        cv2.putText(disp, f"Left→Nose: {int(round(left_nose_avg))} mm", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    else:
        cv2.putText(disp, "Left eye: CLOSED", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    y0 += 28
    if VIEWER_RIGHT_OPEN and right_nose_avg is not None:
        cv2.putText(disp, f"Right→Nose: {int(round(right_nose_avg))} mm", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    else:
        cv2.putText(disp, "Right eye: CLOSED", (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    y0 += 28
    cv2.putText(disp, f"Head tilt: {int(round(head_tilt_deg))}°", (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,200,200) if abs(head_tilt_deg) <= HEAD_TILT_LIMIT_DEG else (0,0,255), 2)

    cv2.putText(disp, "C: Capture  |  Drag pupils to adjust  |  R: Reset auto  |  I: Reset initial  |  Esc: Exit",
                (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # Stream a compact JSON summary for the web overlay
    try:
        # Encode the annotated frame so the web app can render exact pixels
        _, jpg = cv2.imencode('.jpg', disp, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
        out = {
            "pd_mm": int(round(ipd_mm_avg)) if ipd_mm_avg else None,
            "left_to_nose_mm": int(round(left_nose_avg)) if left_nose_avg is not None else None,
            "right_to_nose_mm": int(round(right_nose_avg)) if right_nose_avg is not None else None,
            "head_tilt_deg": int(round(head_tilt_deg)) if head_tilt_deg is not None else None,
            "alignment": bool(alignment_green),
            "distance_mm": int(distance_mm) if distance_mm is not None else None,
            "left_eye": list(final_left) if 'final_left' in locals() and final_left else None,
            "right_eye": list(final_right) if 'final_right' in locals() and final_right else None,
            "frame": {"w": int(w), "h": int(h)},
            "frame_b64": "data:image/jpeg;base64," + b64
        }
        print(json.dumps(out), flush=True)
    except Exception:
        pass

    if not HEADLESS:
        cv2.imshow("Spectacle Eye Detector", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('r'), ord('R')]:
            manual_left_pupil = None
            manual_right_pupil = None
            print("Manual override cleared (reset to latest auto).")
            continue
        if key in [ord('i'), ord('I')]:
            if initial_auto_left_pupil:
                manual_left_pupil = initial_auto_left_pupil
            else:
                manual_left_pupil = None
            if initial_auto_right_pupil:
                manual_right_pupil = initial_auto_right_pupil
            else:
                manual_right_pupil = None
            print("Manual override set to initial auto-detections (if available).")
            continue

        # ---------- Capture with adjustment ----------
        if key in [ord('c'), ord('C')]:
            ts = int(time.time())
            img_name = os.path.join(CAPTURE_SAVE_DIR, f"capture_{ts}.png")
            # Capture the current live cam image for review and saving
            live_disp_clean = disp_clean.copy()
            left_adj, right_adj, result = show_capture_review(
                live_disp_clean, final_left, final_right, scale_to_use, avg_nose_line_point, VIEWER_LEFT_OPEN, VIEWER_RIGHT_OPEN
            )
            if result is True:
                disp_save = live_disp_clean.copy()
                # Draw overlays (oval) on save image (NO AR dot or instruction)
                cv2.ellipse(disp_save, center, (axis_x, axis_y), 0, 0, 360, color_main, 4)
                cv2.ellipse(disp_save, center, (axis_x-6, axis_y-6), 0, 0, 360, (255,255,255), 2)
                # Draw adjusted markers only for open eyes
                if VIEWER_LEFT_OPEN:
                    draw_sniper_cross(disp_save, left_adj, size=8, color=(0,255,0), thickness=1)
                if VIEWER_RIGHT_OPEN:
                    draw_sniper_cross(disp_save, right_adj, size=8, color=(0,255,0), thickness=1)
                # Update readings for saved image
                y0 = 40
                if VIEWER_LEFT_OPEN and VIEWER_RIGHT_OPEN:
                    pd_px = dist(left_adj, right_adj)
                    pd_mm = int(round(pd_px * scale_to_use)) if scale_to_use else 0
                    left_nose_mm = int(round(dist(left_adj, avg_nose_line_point) * scale_to_use)) if avg_nose_line_point else 0
                    right_nose_mm = int(round(dist(right_adj, avg_nose_line_point) * scale_to_use)) if avg_nose_line_point else 0
                    cv2.putText(disp_save, f"PD: {pd_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    y0 += 30
                    cv2.putText(disp_save, f"Left→Nose: {left_nose_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    y0 += 30
                    cv2.putText(disp_save, f"Right→Nose: {right_nose_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                elif VIEWER_LEFT_OPEN:
                    left_nose_mm = int(round(dist(left_adj, avg_nose_line_point) * scale_to_use)) if avg_nose_line_point else 0
                    cv2.putText(disp_save, f"Left→Nose: {left_nose_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    y0 += 30
                    cv2.putText(disp_save, "Right eye: CLOSED", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                elif VIEWER_RIGHT_OPEN:
                    right_nose_mm = int(round(dist(right_adj, avg_nose_line_point) * scale_to_use)) if avg_nose_line_point else 0
                    cv2.putText(disp_save, f"Right→Nose: {right_nose_mm} mm", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    y0 += 30
                    cv2.putText(disp_save, "Left eye: CLOSED", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    cv2.putText(disp_save, "Both eyes: CLOSED", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.imwrite(img_name, disp_save)
                captured_items.append(img_name)
                print(f"Capture saved: {img_name}")
            elif result is False:
                print("Retake capture.")
                continue
            else:
                print("Capture cancelled.")
                continue

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("Captured items:", captured_items)