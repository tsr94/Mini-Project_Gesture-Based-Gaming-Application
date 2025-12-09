import cv2
import mediapipe as mp
import zmq

# ======================
# ZeroMQ Setup
# ======================
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")  # Opens port for Unity to receive

# ======================
# Mediapipe Setup
# ======================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ======================
# Gesture Detection Logic
# ======================
def get_finger_states(landmarks):
    """Return a list of booleans for finger states (extended=True)."""
    tips = [4, 8, 12, 16, 20]
    states = []
    # Thumb
    states.append(landmarks[tips[0]].x < landmarks[tips[0] - 1].x)
    # Other fingers
    for tip in tips[1:]:
        states.append(landmarks[tip].y < landmarks[tip - 2].y)
    return states  # [thumb, index, middle, ring, pinky]

def recognize_gesture(landmarks):
    fingers = get_finger_states(landmarks)
    total_extended = fingers.count(True)

    if total_extended == 0:
        return "SHOOT"  # Fist
    elif total_extended == 5:
        return "STOP"  # Open palm
    elif fingers[1] and not any(fingers[i] for i in [0,2,3,4]):
        return "RELOAD"  # Index only
    elif fingers[1] and fingers[2] and not any(fingers[i] for i in [0,3,4]):
        return "MOVE_FORWARD"  # Two fingers up
    elif fingers[0] and not any(fingers[i] for i in [1,2,3,4]):
        return "JUMP"  # Thumb up
    else:
        return "NONE"

# ======================
# Main Loop
# ======================
cap = cv2.VideoCapture(0)
print("Gesture detection started. Showing webcam...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠ Skipping empty frame")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "NONE"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            gesture = recognize_gesture(hand_landmarks.landmark)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Send gesture command to Unity
    socket.send_string(gesture)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
