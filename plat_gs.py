import cv2
import mediapipe as mp
import zmq

# ======================
# ZeroMQ Setup
# ======================
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

# ======================
# Mediapipe Setup
# ======================
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# ======================
# OpenCV Setup
# ======================
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

def is_hand_open(hand_landmarks):
    """
    Returns True if hand appears open, False if closed.
    Based on whether fingertips are above their corresponding base joints.
    """
    tips = [8, 12, 16, 20]  # Tips of fingers
    bases = [5, 9, 13, 17]  # Base joints of fingers

    open_count = 0
    for tip, base in zip(tips, bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            open_count += 1

    # Hand considered open if 3 or more fingers extended
    if open_count == 2:
        return True
    return open_count >= 3

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    gesture_move = "NONE"
    gesture_jump = "NONE"

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            hand_open = is_hand_open(hand_landmarks)

            # ======================
            # Gesture Logic
            # ======================
            if label == "Right":
                if hand_open:
                    gesture_move = "RIGHT"
                else:
                    gesture_move = "LEFT"

            elif label == "Left":
                if hand_open:
                    gesture_jump = "JUMP"

            # ======================
            # Draw Hand Landmarks
            # ======================
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

    # ======================
    # Combine and Send
    # ======================
    final_gesture = f"{gesture_move}+{gesture_jump}"
    socket.send_string(final_gesture)

    # ======================
    # On-Screen Feedback
    # ======================
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    move_color = (0, 255, 0) if gesture_move != "NONE" else (200, 200, 200)
    jump_color = (0, 255, 255) if gesture_jump != "NONE" else (200, 200, 200)

    cv2.putText(frame, f"Move: {gesture_move}", (20, 40), font, 1, move_color, 2)
    cv2.putText(frame, f"Jump: {gesture_jump}", (300, 40), font, 1, jump_color, 2)

    cv2.imshow("Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
