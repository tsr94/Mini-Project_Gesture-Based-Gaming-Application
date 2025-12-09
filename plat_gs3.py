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


def count_extended_fingers(hand_landmarks):
    """
    Returns number of extended fingers (excluding thumb for simplicity)
    """
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    bases = [5, 9, 13, 17]  # Corresponding base joints

    count = 0
    for tip, base in zip(tips, bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            count += 1
    return count


def draw_legend(frame, gesture_move, gesture_jump):
    """
    Draws a small compact legend in the bottom-right corner.
    """
    h, w, _ = frame.shape

    # Small box size
    box_w = 230
    box_h = 80
    pad = 8

    # bottom-right position
    x0 = w - box_w - 10
    y0 = h - box_h - 10

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "Gestures", (x0 + pad, y0 + 22), font, 0.55, (255, 255, 255), 2)

    # Lines
    cv2.putText(frame, "Peace - Move RIGHT", (x0 + pad, y0 + 45),
                font, 0.45, (0, 255, 0) if gesture_move == "RIGHT" else (200, 200, 200), 1)

    cv2.putText(frame, "Fist Right - Move LEFT", (x0 + pad, y0 + 62),
                font, 0.45, (0, 255, 0) if gesture_move == "LEFT" else (200, 200, 200), 1)

    cv2.putText(frame, "Open Left - JUMP", (x0 + pad, y0 + 79),
                font, 0.45, (0, 255, 255) if gesture_jump == "JUMP" else (200, 200, 200), 1)


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
            extended = count_extended_fingers(hand_landmarks)

            # ======================
            # RIGHT HAND LOGIC
            # ======================
            if label == "Right":
                if extended == 2:  # Peace sign (two fingers)
                    gesture_move = "RIGHT"
                elif extended == 0:  # Closed fist
                    gesture_move = "LEFT"
                elif extended >= 4:  # Open palm
                    gesture_move = "NONE"

            # ======================
            # LEFT HAND LOGIC
            # ======================
            elif label == "Left":
                if extended >= 4:  # Open palm
                    gesture_jump = "JUMP"
                elif extended == 0:  # Closed fist
                    gesture_jump = "NONE"

            # ======================
            # Draw landmarks
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
    # On-Screen Feedback (top bar)
    # ======================
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    move_color = (0, 255, 0) if gesture_move != "NONE" else (200, 200, 200)
    jump_color = (0, 255, 255) if gesture_jump != "NONE" else (200, 200, 200)

    cv2.putText(frame, f"Move: {gesture_move}", (20, 40), font, 1, move_color, 2)
    cv2.putText(frame, f"Jump: {gesture_jump}", (300, 40), font, 1, jump_color, 2)

    # ======================
    # Draw the legend / guide
    # ======================
    draw_legend(frame, gesture_move, gesture_jump)

    cv2.imshow("Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
