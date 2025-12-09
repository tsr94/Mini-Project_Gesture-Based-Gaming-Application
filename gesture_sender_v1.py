import cv2
import mediapipe as mp
import zmq
import math
import time

# ======================
# Config
# ======================
DEBUG = False  # set True to print finger arrays and handedness to console

# Look control tuning (tweak these)
LOOK_SENSITIVITY = 0.8        # larger = bigger camera movement for same eye offset
LOOK_SMOOTHING = 0.88        # 0..1, higher = smoother (slower). Use ~0.6-0.85
LOOK_DEADZONE = 0.01      # normalized coordinate deadzone for tiny jitter (lower tolerance = more sensitive)
STILL_FRAMES_TO_RESET = 6    # how many consecutive frames inside deadzone before resetting neutral
LOOK_CLAMP = 0.2            # max absolute look value before clamping (final values will be clamped again later)
output_scale = 0.08
# ======================
# ZeroMQ Setup (unchanged)
# ======================
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")  # Unity can connect to tcp://<your_ip>:5555

# ======================
# Mediapipe Setup
# ======================
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ======================
# Helpers: finger + gesture detection (unchanged)
# ======================
def get_finger_states(landmarks, handedness_label="Right"):
    tips = [4, 8, 12, 16, 20]
    states = [False] * 5

    tip = landmarks[tips[0]]
    ip = landmarks[tips[0] - 1]
    mcp = landmarks[tips[0] - 2]

    if handedness_label == "Right":
        states[0] = (tip.x < ip.x) or (tip.x < mcp.x)
    else:
        states[0] = (tip.x > ip.x) or (tip.x > mcp.x)

    for i, tip_idx in enumerate(tips[1:], start=1):
        tip_l = landmarks[tip_idx]
        pip_l = landmarks[tip_idx - 2]
        states[i] = (tip_l.y < pip_l.y - 0.01)

    return states


def recognize_gesture(landmarks, handedness_label="Right"):
    fingers = get_finger_states(landmarks, handedness_label=handedness_label)
    total_extended = fingers.count(True)

    if total_extended == 0:
        return "SHOOT"
    elif total_extended == 5:
        return "STOP"
    elif fingers[0] and not any(fingers[i] for i in [1,2,3,4]):
        return "JUMP"
    else:
        return "NONE"


def detect_movement(left_hand, right_hand, left_label=None, right_label=None):
    movement = "NONE"

    if left_hand:
        left_fingers = get_finger_states(left_hand.landmark, handedness_label=(left_label or "Left"))
        left_total_extended = sum(1 for f in left_fingers if f)
        if DEBUG:
            print("Left label:", left_label, "Left fingers:", left_fingers, "Total:", left_total_extended)

        if left_total_extended >= 4:
            movement = "MOVE_FORWARD"
        elif left_fingers[1] and not any(left_fingers[i] for i in [2,3,4]):
            movement = "MOVE_LEFT"
        elif left_total_extended <= 1:
            movement = "MOVE_BACKWARD"

    if right_hand:
        right_fingers = get_finger_states(right_hand.landmark, handedness_label=(right_label or "Right"))
        if DEBUG:
            print("Right label:", right_label, "Right fingers:", right_fingers)
        if right_fingers[1] and not any(right_fingers[i] for i in [0,2,3,4]):
            movement = "MOVE_RIGHT"

    return movement

# ======================
# Look Control (new approach)
# ======================
# State for look control
neutral_face_pos = None  # neutral gaze center (x,y)
ema_dx = 0.0             # exponentially smoothed dx returned to Unity
ema_dy = 0.0
still_counter = 0
prev_look_enabled = False

def get_face_look_delta(face_landmarks, look_enabled):
    """
    Convert face_landmarks to a (dx, dy) look vector in normalized units.
    - Uses absolute offset from a neutral gaze center (neutral_face_pos).
    - Applies sensitivity, EMA smoothing, deadzone and reset when still.
    - When look_enabled transitions from False->True, neutral is initialized
      to the current gaze to avoid jumps.
    """
    global neutral_face_pos, ema_dx, ema_dy, still_counter, prev_look_enabled

    # Basic checks
    if not face_landmarks or len(face_landmarks) < 478:
        # No face -> forget neutral so re-init later
        neutral_face_pos = None
        ema_dx = 0.0
        ema_dy = 0.0
        still_counter = 0
        prev_look_enabled = look_enabled
        return 0.0, 0.0

    # Compute gaze center using iris landmarks
    left_iris = face_landmarks[468]
    right_iris = face_landmarks[473]
    gaze_x = (left_iris.x + right_iris.x) / 2.0
    gaze_y = (left_iris.y + right_iris.y) / 2.0

    # Initialize neutral when enabling look or if neutral missing
    if neutral_face_pos is None or (look_enabled and not prev_look_enabled):
        neutral_face_pos = (gaze_x, gaze_y)
        ema_dx = 0.0
        ema_dy = 0.0
        still_counter = 0
        prev_look_enabled = look_enabled
        return 0.0, 0.0

    prev_look_enabled = look_enabled

    # Absolute offset from neutral
    raw_dx = gaze_x - neutral_face_pos[0]
    raw_dy = neutral_face_pos[1] - gaze_y   # invert y so up = positive

    # Deadzone check (tiny jitter ignored)
    if abs(raw_dx) < LOOK_DEADZONE and abs(raw_dy) < LOOK_DEADZONE:
        still_counter += 1
    else:
        still_counter = 0

    # If the gaze is "still" for enough frames, reset neutral to current gaze
    if still_counter >= STILL_FRAMES_TO_RESET:
        neutral_face_pos = (gaze_x, gaze_y)
        ema_dx = 0.0
        ema_dy = 0.0
        still_counter = 0
        if DEBUG:
            print("[LOOK] neutral reset due to stillness")
        return 0.0, 0.0

    # Scale by sensitivity to get target camera movement
    target_dx = raw_dx * LOOK_SENSITIVITY
    target_dy = raw_dy * LOOK_SENSITIVITY

    # Clamp raw target to reasonable bounds (prevents wild jumps)
    target_dx = max(-LOOK_CLAMP, min(LOOK_CLAMP, target_dx))
    target_dy = max(-LOOK_CLAMP, min(LOOK_CLAMP, target_dy))

    # Exponential smoothing (EMA)
    alpha = 1.0 - LOOK_SMOOTHING  # smoothing factor for new sample
    ema_dx = ema_dx * LOOK_SMOOTHING + target_dx * alpha
    ema_dy = ema_dy * LOOK_SMOOTHING + target_dy * alpha

    # Final clamp to stricter range
    ema_dx = max(-0.5, min(0.5, ema_dx))
    ema_dy = max(-0.5, min(0.5, ema_dy))

    if DEBUG:
        print(f"[LOOK] gaze=({gaze_x:.4f},{gaze_y:.4f}) neutral=({neutral_face_pos[0]:.4f},{neutral_face_pos[1]:.4f}) raw=({raw_dx:.4f},{raw_dy:.4f}) target=({target_dx:.3f},{target_dy:.3f}) ema=({ema_dx:.3f},{ema_dy:.3f}) still={still_counter}")

    return ema_dx, ema_dy

# ======================
# Main Loop (full)
# ======================
cap = cv2.VideoCapture(0)
print("Gesture detection started. Showing webcam...")
look_enabled = False  # toggled by STOP (open palm) / SHOOT (fist)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠ Skipping empty frame")
        continue

    # Mirror for user (we will process the mirrored image so MediaPipe handedness refers to displayed image)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands and face
    results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    gesture = "NONE"
    movement = "NONE"
    look_dx, look_dy = 0.0, 0.0
    left_hand = None
    right_hand = None
    left_label = None
    right_label = None

    # HANDS: find left and right (MediaPipe labels refer to the processed image)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            if label == "Left":
                left_hand = hand_landmarks
                left_label = label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
            else:
                right_hand = hand_landmarks
                right_label = label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

        # Determine gesture from right hand (if present)
        if right_hand:
            gesture = recognize_gesture(right_hand.landmark, handedness_label=(right_label or "Right"))

            # Gesture-based toggle for look control
            if gesture == "STOP":      # Open palm → start look
                look_enabled = True
            elif gesture == "SHOOT":   # Fist → stop look
                look_enabled = False

        # Movement detection using both hands (left-hand priority)
        movement = detect_movement(left_hand, right_hand, left_label=left_label, right_label=right_label)

    # FACE: look delta calculation using improved method
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        # Only compute look delta if look control is enabled and gesture allowed it
        if look_enabled and right_hand:
            look_dx, look_dy = get_face_look_delta(face_landmarks, look_enabled=True)
        else:
            # If look is disabled, we want to reset look state so enabling later won't jump
            # Call get_face_look_delta with look_enabled=False to clear if face disappears
            _ = get_face_look_delta(face_landmarks, look_enabled=False)
            look_dx, look_dy = 0.0, 0.0

        # Draw iris markers on the display frame
        h, w, _ = frame.shape
        left_eye = face_landmarks[468]
        right_eye = face_landmarks[473]
        left_eye_pos = (int(left_eye.x * w), int(left_eye.y * h))
        right_eye_pos = (int(right_eye.x * w), int(right_eye.y * h))
        cv2.circle(frame, left_eye_pos, 3, (255, 0, 255), -1)
        cv2.circle(frame, right_eye_pos, 3, (255, 0, 255), -1)

        # Display status lines (no overlap)
        cv2.putText(frame, "Face Tracking Active", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if look_enabled:
            cv2.putText(frame, "Eye Tracking Active (LOOK ON)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        else:
            cv2.putText(frame, "Eye Tracking Inactive (LOOK OFF)", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Prepare and send command to Unity
    if movement == "STOP":
        movement = "NONE"
    scaled_dx = look_dx * output_scale
    scaled_dy = look_dy * output_scale

    command = f"{gesture}|{movement}|{scaled_dx:.3f},{scaled_dy:.3f}"
    socket.send_string(command)

    # On-screen debug text (well spaced)
    cv2.putText(frame, f"Right Hand (Gesture): {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Left Hand (Movement): {movement}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Command: {command}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Finger state debug (small, beneath)
    if left_hand:
        left_fingers = get_finger_states(left_hand.landmark, handedness_label=(left_label or "Left"))
        cv2.putText(frame, f"Left Fingers: {[1 if f else 0 for f in left_fingers]}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if right_hand:
        right_fingers = get_finger_states(right_hand.landmark, handedness_label=(right_label or "Right"))
        cv2.putText(frame, f"Right Fingers: {[1 if f else 0 for f in right_fingers]}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show numeric look deltas on screen for feedback
    cv2.putText(frame, f"Look Delta: {look_dx:.3f}, {look_dy:.3f}", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,255), 2)

    cv2.imshow("Gesture & Movement Detection", frame)

    if DEBUG:
        print(f"Gesture:{gesture} Movement:{movement} Look:{look_enabled} Delta:({look_dx:.3f},{look_dy:.3f})")

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
