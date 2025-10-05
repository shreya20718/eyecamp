"""Integrated Spectacle app

Tabs: "Pupil Detector" and "Frame Fitter"
- reserves a top tab bar so image content is never covered
- forwards mouse events to active tool (y translated by TAB_BAR_HEIGHT)

This file depends on OpenCV, NumPy and MediaPipe (same as original scripts).
"""
import os
import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# Import SpectacleFitterTool from v_FrameDetection if available
try:
    from v_FrameDetection import SpectacleFitterTool
except Exception:
    # try relative import fallback
    try:
        from .v_FrameDetection import SpectacleFitterTool
    except Exception:
        SpectacleFitterTool = None

TAB_BAR_HEIGHT = 64


class PupilDetector:
    """A compact pupil detector module used inside the integrated app.

    This is a simplified adaptation of the larger v_PupilDetection script focusing on
    retaining the UI and mouse/button behaviors so it can run inside the tabbed window.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True,
                                                   max_num_faces=1,
                                                   min_detection_confidence=0.5,
                                                   min_tracking_confidence=0.5)

        self.manual_left_pupil = None
        self.manual_right_pupil = None
        self.selected_eye = None

        self.auto_left_pupil = None
        self.auto_right_pupil = None
        self.initial_auto_left_pupil = None
        self.initial_auto_right_pupil = None

        self.last_frame_shape = None
        self.capture_requested = False

    def dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1]) if a and b else 0.0

    def get_button_layout(self, frame_shape):
        h, w = frame_shape[0], frame_shape[1]
        buttons = {}
        margin = 12
        btn_h = 36
        btn_w = 140
        x = margin
        by = h - margin - btn_h
        order = [
            ('btn_capture', 'Capture'),
            ('btn_reset', 'Reset'),
            ('btn_reset_init', 'ResetInit'),
            ('btn_quit', 'Quit')
        ]
        for key, label in order:
            buttons[key] = (x, by, btn_w, btn_h, label)
            x += btn_w + 12
        return buttons

    def draw_buttons(self, frame):
        if frame is None:
            return {}
        h, w = frame.shape[:2]
        self.last_frame_shape = (h, w)
        overlay = frame.copy()
        bar_h = 64
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        layout = self.get_button_layout((h, w))
        for key, (bx, by, bw, bh, label) in layout.items():
            btn_color = (40, 40, 40)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), btn_color, -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (220, 220, 220), 2)
            text_x = bx + 12
            text_y = by + bh//2 + 6
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245,245,245), 2)
        return layout

    def handle_button_click(self, x, y):
        if self.last_frame_shape is None:
            return False
        layout = self.get_button_layout(self.last_frame_shape)
        for key, (bx, by, bw, bh, label) in layout.items():
            if x >= bx and x <= bx + bw and y >= by and y <= by + bh:
                if key == 'btn_capture':
                    self.capture_requested = True
                    return True
                if key == 'btn_reset':
                    self.manual_left_pupil = None
                    self.manual_right_pupil = None
                    return True
                if key == 'btn_reset_init':
                    if self.initial_auto_left_pupil:
                        self.manual_left_pupil = self.initial_auto_left_pupil
                    if self.initial_auto_right_pupil:
                        self.manual_right_pupil = self.initial_auto_right_pupil
                    return True
                if key == 'btn_quit':
                    return True
        return False

    def mouse_callback(self, event, x, y, flags, param):
        # x,y already translated by TAB_BAR_HEIGHT in the integrator
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.handle_button_click(x, y):
                return
            if self.auto_left_pupil and self.dist((x, y), self.auto_left_pupil) < 30:
                self.selected_eye = 'left'
                self.manual_left_pupil = self.auto_left_pupil
            elif self.auto_right_pupil and self.dist((x, y), self.auto_right_pupil) < 30:
                self.selected_eye = 'right'
                self.manual_right_pupil = self.auto_right_pupil

        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self.selected_eye == 'left':
                self.manual_left_pupil = (x, y)
            elif self.selected_eye == 'right':
                self.manual_right_pupil = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_eye = None

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        self.auto_left_pupil = None
        self.auto_right_pupil = None

        pixels_per_mm = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            try:
                left_idxs = list(range(468, 474))
                right_idxs = list(range(473, 479))
                lpts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in left_idxs if i < len(lm)]
                rpts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in right_idxs if i < len(lm)]
                if lpts:
                    self.auto_left_pupil = (int(sum([p[0] for p in lpts])/len(lpts)), int(sum([p[1] for p in lpts])/len(lpts)))
                    if self.initial_auto_left_pupil is None:
                        self.initial_auto_left_pupil = self.auto_left_pupil
                if rpts:
                    self.auto_right_pupil = (int(sum([p[0] for p in rpts])/len(rpts)), int(sum([p[1] for p in rpts])/len(rpts)))
                    if self.initial_auto_right_pupil is None:
                        self.initial_auto_right_pupil = self.auto_right_pupil
                if len(lm) > 362:
                    left_inner = (int(lm[133].x * w), int(lm[133].y * h))
                    right_inner = (int(lm[362].x * w), int(lm[362].y * h))
                    ipd_px = self.dist(left_inner, right_inner)
                    if ipd_px > 0:
                        pixels_per_mm = ipd_px / 63.0
            except Exception:
                pixels_per_mm = None

        final_left = self.manual_left_pupil if self.manual_left_pupil is not None else self.auto_left_pupil
        final_right = self.manual_right_pupil if self.manual_right_pupil is not None else self.auto_right_pupil

        disp = frame.copy()
        if final_left:
            cv2.drawMarker(disp, final_left, (0,255,0), cv2.MARKER_CROSS, 8, 1)
        if final_right:
            cv2.drawMarker(disp, final_right, (0,255,0), cv2.MARKER_CROSS, 8, 1)

        if final_left and final_right and pixels_per_mm:
            pd_px = self.dist(final_left, final_right)
            pd_mm = int(round(pd_px / pixels_per_mm))
            cv2.putText(disp, f"PD: {pd_mm} mm", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        self.draw_buttons(disp)
        return disp


def draw_tab_bar(frame, active_tab_index, tab_names):
    h, w = frame.shape[:2]
    bar = np.zeros((TAB_BAR_HEIGHT, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)
    # draw tabs
    margin = 8
    tab_w = 200
    x = margin
    for i, name in enumerate(tab_names):
        color = (60, 160, 60) if i == active_tab_index else (100, 100, 100)
        cv2.rectangle(bar, (x, 8), (x + tab_w, TAB_BAR_HEIGHT - 12), color, -1)
        cv2.rectangle(bar, (x, 8), (x + tab_w, TAB_BAR_HEIGHT - 12), (220,220,220), 1)
        txt_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx = x + (tab_w - txt_size[0]) // 2
        ty = (TAB_BAR_HEIGHT // 2) + (txt_size[1] // 2)
        cv2.putText(bar, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        x += tab_w + margin
    return bar


def main():
    pupil = PupilDetector()
    fitter = SpectacleFitterTool() if SpectacleFitterTool is not None else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    window_name = 'OptiFocus - Integrated'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    tab_names = ['Pupil Detector', 'Frame Fitter']
    active_tab = 0

    # mouse router
    def on_mouse(event, x, y, flags, param):
        nonlocal active_tab
        # if click happened inside tab bar - check tab switches
        if y <= TAB_BAR_HEIGHT:
            # detect which tab clicked
            # tabs are drawn starting at margin 8 and width 200
            margin = 8
            tab_w = 200
            idx = (x - margin) // (tab_w + margin)
            if 0 <= idx < len(tab_names):
                active_tab = int(idx)
            return

        # translate y by removing tab bar before forwarding to tools
        ty = y - TAB_BAR_HEIGHT
        # forward to active tool
        if active_tab == 0:
            pupil.mouse_callback(event, x, ty, flags, None)
        elif active_tab == 1 and fitter is not None:
            fitter.mouse_callback(event, x, ty, flags, None)

    cv2.setMouseCallback(window_name, on_mouse)

    print("Integrated app starting. Tabs: ", tab_names)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # prepare canvas taller by TAB_BAR_HEIGHT so tabs don't cover content
        canvas = np.zeros((h + TAB_BAR_HEIGHT, w, 3), dtype=np.uint8)
        # compute displayed area for tool (y offset TAB_BAR_HEIGHT)
        tool_area = canvas[TAB_BAR_HEIGHT:TAB_BAR_HEIGHT + h, 0:w]
        # copy frame into tool area
        tool_area[:] = frame

        if active_tab == 0:
            disp = pupil.process_frame(tool_area.copy())
            # remember last displayed annotated frame for pupil tool and handle capture
            try:
                pupil.last_display_frame = disp.copy()
            except Exception:
                pupil.last_display_frame = None
            if getattr(pupil, 'capture_requested', False):
                # reset flag and save current annotated frame
                pupil.capture_requested = False
                os.makedirs("./captures", exist_ok=True)
                ts = int(time.time())
                img_name = os.path.join("./captures", f"pupil_capture_{ts}.png")
                try:
                    if pupil.last_display_frame is not None:
                        cv2.imwrite(img_name, pupil.last_display_frame)
                        print(f"Capture saved: {img_name}")
                    else:
                        print("No frame available to save for pupil capture.")
                except Exception as e:
                    print("Failed to save capture:", e)
        elif active_tab == 1 and fitter is not None:
            # fitter expects full image; we forward the tool_area (no tab)
            fitter.landmarks_data = fitter.get_face_landmarks(tool_area)
            if fitter.landmarks_data and fitter.pixels_per_mm is None:
                fitter.calibrate(fitter.landmarks_data, tool_area.shape)
            if fitter.landmarks_data and not fitter.initial_lines_set and fitter.pixels_per_mm is not None:
                fitter.setup_initial_lines(fitter.landmarks_data, tool_area)
            disp = fitter.draw_all(tool_area.copy())
            # store last displayed annotated frame so capture button can save it
            try:
                fitter.last_display_frame = disp.copy()
            except Exception:
                fitter.last_display_frame = None
        else:
            disp = tool_area.copy()

        # place disp back into canvas
        canvas[TAB_BAR_HEIGHT:TAB_BAR_HEIGHT + h, 0:w] = disp
        # draw tab bar
        tabbar = draw_tab_bar(canvas, active_tab, tab_names)
        canvas[0:TAB_BAR_HEIGHT, 0:w] = tabbar

        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import os
from collections import deque


# Integrated app: two tabs (Pupil Detector, Frame Fitter)
# - reserves a top tab bar so image content is never covered
# - forwards mouse events to active tool (y translated by TAB_BAR_HEIGHT)

TAB_BAR_HEIGHT = 64


class PupilDetector:
    """Simplified pupil detector based on the v_PupilDetection script.
    Draws the same bottom buttons and supports basic mouse dragging for pupil markers.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True,
                                                  max_num_faces=1,
                                                  min_detection_confidence=0.5,
                                                  min_tracking_confidence=0.5)

        self.iris_real_mm = 11.7

        import cv2
        import numpy as np
        import mediapipe as mp
        import math
        import time
        import os
        from collections import deque


        # Integrated app: two tabs (Pupil Detector, Frame Fitter)
        # - reserves a top tab bar so image content is never covered
        # - forwards mouse events to active tool (y translated by TAB_BAR_HEIGHT)

        TAB_BAR_HEIGHT = 64


        class PupilDetector:
            """Simplified pupil detector based on the v_PupilDetection script.
            Draws the same bottom buttons and supports basic mouse dragging for pupil markers.
            """
            def __init__(self):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True,
                                                           max_num_faces=1,
                                                           min_detection_confidence=0.5,
                                                           min_tracking_confidence=0.5)

                self.iris_real_mm = 11.7
                self.ipd_history = deque(maxlen=15)

                self.manual_left_pupil = None
                self.manual_right_pupil = None
                self.selected_eye = None

                self.auto_left_pupil = None
                self.auto_right_pupil = None
                self.initial_auto_left_pupil = None
                self.initial_auto_right_pupil = None

                self.last_display_frame = None
                self.last_frame_shape = None
                self.capture_requested = False

            def dist(self, a, b):
                return math.hypot(a[0]-b[0], a[1]-b[1]) if a and b else 0.0

            def get_button_layout(self, frame_shape):
                h, w = frame_shape[0], frame_shape[1]
                buttons = {}
                margin = 12
                btn_h = 36
                btn_w = 140
                x = margin
                by = h - margin - btn_h
                order = [
                    ('btn_capture', 'Capture'),
                    ('btn_reset', 'Reset'),
                    ('btn_reset_init', 'ResetInit'),
                    ('btn_quit', 'Quit')
                ]
                for key, label in order:
                    buttons[key] = (x, by, btn_w, btn_h, label)
                    x += btn_w + 12
                return buttons

            def draw_buttons(self, frame):
                if frame is None:
                    return {}
                h, w = frame.shape[:2]
                self.last_frame_shape = (h, w)
                overlay = frame.copy()
                bar_h = 64
                cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
                alpha = 0.35
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                layout = self.get_button_layout((h, w))
                for key, (bx, by, bw, bh, label) in layout.items():
                    btn_color = (40, 40, 40)
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), btn_color, -1)
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (220, 220, 220), 2)
                    text_x = bx + 12
                    text_y = by + bh//2 + 6
                    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245,245,245), 2)
                return layout

            def handle_button_click(self, x, y):
                if self.last_frame_shape is None:
                    return False
                layout = self.get_button_layout(self.last_frame_shape)
                for key, (bx, by, bw, bh, label) in layout.items():
                    if x >= bx and x <= bx + bw and y >= by and y <= by + bh:
                        if key == 'btn_capture':
                            self.capture_requested = True
                            return True
                        if key == 'btn_reset':
                            self.manual_left_pupil = None
                            self.manual_right_pupil = None
                            return True
                        if key == 'btn_reset_init':
                            if self.initial_auto_left_pupil:
                                self.manual_left_pupil = self.initial_auto_left_pupil
                            if self.initial_auto_right_pupil:
                                self.manual_right_pupil = self.initial_auto_right_pupil
                            return True
                        if key == 'btn_quit':
                            # handled by main app
                            return True
                return False

            def mouse_callback(self, event, x, y, flags, param):
                # x,y already translated by TAB_BAR_HEIGHT in the integrator
                if event == cv2.EVENT_LBUTTONDOWN:
                    if self.handle_button_click(x, y):
                        return
                    # click near whichever pupil
                    if self.auto_left_pupil and self.dist((x, y), self.auto_left_pupil) < 30:
                        self.selected_eye = 'left'
                        self.manual_left_pupil = self.auto_left_pupil
                    elif self.auto_right_pupil and self.dist((x, y), self.auto_right_pupil) < 30:
                        self.selected_eye = 'right'
                        self.manual_right_pupil = self.auto_right_pupil

                elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                    if self.selected_eye == 'left':
                        self.manual_left_pupil = (x, y)
                    elif self.selected_eye == 'right':
                        self.manual_right_pupil = (x, y)
                elif event == cv2.EVENT_LBUTTONUP:
                    self.selected_eye = None

            def process_frame(self, frame):
                # returns annotated frame to be shown (image size unchanged)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                self.auto_left_pupil = None
                self.auto_right_pupil = None

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    # compute simple pupil centers by averaging iris landmarks if present
                    # Mediapipe iris landmarks usually 468-473 (left) and 473-478 (right) — we'll attempt to use ranges
                    try:
                        left_idxs = list(range(468, 474))
                        right_idxs = list(range(473, 479))
                        lpts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in left_idxs if i < len(lm)]
                        rpts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in right_idxs if i < len(lm)]
                        if lpts:
                            self.auto_left_pupil = (int(sum([p[0] for p in lpts])/len(lpts)), int(sum([p[1] for p in lpts])/len(lpts)))
                            if self.initial_auto_left_pupil is None:
                                self.initial_auto_left_pupil = self.auto_left_pupil
                        if rpts:
                            self.auto_right_pupil = (int(sum([p[0] for p in rpts])/len(rpts)), int(sum([p[1] for p in rpts])/len(rpts)))
                            if self.initial_auto_right_pupil is None:
                                self.initial_auto_right_pupil = self.auto_right_pupil
                        # compute IPD scale using inner eye landmarks (133, 362)
                        if len(lm) > 362:
                            left_inner = (int(lm[133].x * w), int(lm[133].y * h))
                            right_inner = (int(lm[362].x * w), int(lm[362].y * h))
                            ipd_px = self.dist(left_inner, right_inner)
                            if ipd_px > 0:
                                pixels_per_mm = ipd_px / 63.0
                            else:
                                pixels_per_mm = None
                        else:
                            pixels_per_mm = None
                    except Exception:
                        pixels_per_mm = None

                final_left = self.manual_left_pupil if self.manual_left_pupil is not None else self.auto_left_pupil
                final_right = self.manual_right_pupil if self.manual_right_pupil is not None else self.auto_right_pupil

                disp = frame.copy()
                # draw markers
                if final_left:
                    cv2.drawMarker(disp, final_left, (0,255,0), cv2.MARKER_CROSS, 8, 1)
                if final_right:
                    cv2.drawMarker(disp, final_right, (0,255,0), cv2.MARKER_CROSS, 8, 1)

                # show PD
                if final_left and final_right and pixels_per_mm:
                    pd_px = self.dist(final_left, final_right)
                    pd_mm = int(round(pd_px / pixels_per_mm))
                    cv2.putText(disp, f"PD: {pd_mm} mm", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                # draw bottom buttons
                self.draw_buttons(disp)
                self.last_display_frame = disp.copy()
                return disp

