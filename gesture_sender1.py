import cv2
import mediapipe as mp
import zmq
import math
import time

# ======================
# Config
# ======================
DEBUG = False  # set True to print finger arrays and handedness to console

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
# Helpers: finger + gesture detection
# ======================
def get_finger_states(landmarks, handedness_label="Right"):
    """
    landmarks: mp hand landmarks list-like (indexable)
    handedness_label: "Left" or "Right" as reported by MediaPipe for the processed image
    Returns: [thumb, index, middle, ring, pinky] booleans
    """
    tips = [4, 8, 12, 16, 20]
    states = [False] * 5

    # Thumb: compare TIP (4) with IP/MCP, but direction depends on handedness
    tip = landmarks[tips[0]]
    ip = landmarks[tips[0] - 1]   # 3
    mcp = landmarks[tips[0] - 2]  # 2

    # Use OR checks for robustness, small margin not required for x here
    if handedness_label == "Right":
        # For right hand in the processed (possibly flipped) image, tip.x < ip.x means thumb extended (to the left)
        states[0] = (tip.x < ip.x) or (tip.x < mcp.x)
    else:  # Left hand
        states[0] = (tip.x > ip.x) or (tip.x > mcp.x)

    # Other fingers: tip.y < pip.y indicates extended (y=0 top of image in normalized coords)
    for i, tip_idx in enumerate(tips[1:], start=1):
        tip_l = landmarks[tip_idx]
        pip_l = landmarks[tip_idx - 2]  # pip joint indices: 6,10,14,18
        # small margin to avoid jitter
        states[i] = (tip_l.y < pip_l.y - 0.01)

    return states  # [thumb, index, middle, ring, pinky]


def recognize_gesture(landmarks, handedness_label="Right"):
    """Return gesture string from single-hand landmarks and handedness_label."""
    fingers = get_finger_states(landmarks, handedness_label=handedness_label)
    total_extended = fingers.count(True)

    # Fist
    if total_extended == 0:
        return "SHOOT"
    # Open palm
    elif total_extended == 5:
        return "STOP"
    # Thumb up (thumb extended, others not)
    elif fingers[0] and not any(fingers[i] for i in [1,2,3,4]):
        return "JUMP"
    else:
        return "NONE"


def detect_movement(left_hand, right_hand, left_label=None, right_label=None):
    """
    Detect movement using left and right hand landmarks and their labels.
    left_hand/right_hand should be mp hand landmark objects or None.
    left_label/right_label should be the corresponding "Left"/"Right" strings or None.
    Returns movement string: MOVE_FORWARD, MOVE_LEFT, MOVE_BACKWARD, MOVE_RIGHT, or NONE
    """
    movement = "NONE"

    # Left-hand movement detection (priority)
    if left_hand:
        left_fingers = get_finger_states(left_hand.landmark, handedness_label=(left_label or "Left"))
        left_total_extended = sum(1 for f in left_fingers if f)
        if DEBUG:
            print("Left label:", left_label, "Left fingers:", left_fingers, "Total:", left_total_extended)

        # Open palm -> move forward (robust: >= 4)
        if left_total_extended >= 4:
            movement = "MOVE_FORWARD"
        # Index only -> move left
        elif left_fingers[1] and not any(left_fingers[i] for i in [2,3,4]):
            movement = "MOVE_LEFT"
        # Fist/closed -> move backward (<=1 extended)
        elif left_total_extended <= 1:
            movement = "MOVE_BACKWARD"

    # Right-hand specific: if index-only -> move right (overrides left unless more specific left movement required)
    if right_hand:
        right_fingers = get_finger_states(right_hand.landmark, handedness_label=(right_label or "Right"))
        if DEBUG:
            print("Right label:", right_label, "Right fingers:", right_fingers)
        if right_fingers[1] and not any(right_fingers[i] for i in [0,2,3,4]):
            movement = "MOVE_RIGHT"

    return movement


# ======================
# Look Control (Eye delta) - unchanged logic mostly, only small tweak
# ======================
prev_face_pos = None

def get_face_look_delta(face_landmarks):
    """Compute camera look delta using iris centers. Returns (dx, dy)."""
    global prev_face_pos

    if not face_landmarks or len(face_landmarks) < 478:
        prev_face_pos = None
        return (0.0, 0.0)

    left_iris = face_landmarks[468]
    right_iris = face_landmarks[473]

    x = (left_iris.x + right_iris.x) / 2
    y = (left_iris.y + right_iris.y) / 2

    dx, dy = 0.0, 0.0
    if prev_face_pos:
        dx = x - prev_face_pos[0]
        dy = prev_face_pos[1] - y

    prev_face_pos = (x, y)

    # Dead zone
    if abs(dx) < 0.001: dx = 0.0
    if abs(dy) < 0.001: dy = 0.0

    sensitivity = 8.0
    smooth_factor = 0.75

    dx *= sensitivity * (1 - smooth_factor)
    dy *= sensitivity * (1 - smooth_factor)

    dx = max(-0.5, min(0.5, dx))
    dy = max(-0.5, min(0.5, dy))

    return (dx, dy)


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

    # FACE: look delta calculation
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        # Only compute look delta if look control is enabled and gesture allowed it
        if look_enabled and right_hand:
            look_dx, look_dy = get_face_look_delta(face_landmarks)
        else:
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
    command = f"{gesture}|{movement}|{look_dx:.3f},{look_dy:.3f}"
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

    cv2.imshow("Gesture & Movement Detection", frame)

    if DEBUG:
        # Print one-line summary for quick debugging
        print(f"Gesture:{gesture} Movement:{movement} Look:{look_enabled} Delta:({look_dx:.3f},{look_dy:.3f})")

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
