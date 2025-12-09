import cv2
import mediapipe as mp
import zmq
import time
import math

# ==== ZeroMQ ====
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")
print("[Python] Publisher started on tcp://*:5555")

# ==== Mediapipe ====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# ==== Camera ====
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)

last_command = ""
last_sent_time = 0
cooldown = 0.3

def send_command(command):
    global last_command, last_sent_time
    now = time.time()
    if command != last_command or (now - last_sent_time) > cooldown:
        socket.send_string(command)
        print(f"[Python] Sent: {command}")
        last_command = command
        last_sent_time = now

# ==== Helper ====
def finger_extended(hand, tip_id):
    return hand.landmark[tip_id].y < hand.landmark[tip_id - 2].y

def get_extended_fingers(hand):
    tips = [8, 12, 16, 20]
    return [finger_extended(hand, t) for t in tips]

def is_fist(hand):
    return sum(get_extended_fingers(hand)) <= 1

def is_open_palm(hand):
    return sum(get_extended_fingers(hand)) >= 3

def only_index(hand):
    fingers = get_extended_fingers(hand)
    return fingers[0] and not any(fingers[1:])

def index_middle(hand):
    fingers = get_extended_fingers(hand)
    return fingers[0] and fingers[1] and not any(fingers[2:])

def thumb_pinky(hand):
    thumb_tip = hand.landmark[4]
    pinky_tip = hand.landmark[20]
    thumb_mcp = hand.landmark[2]
    pinky_mcp = hand.landmark[17]
    return (thumb_tip.y < thumb_mcp.y) and (pinky_tip.y < pinky_mcp.y)

def get_wrist_angle(hand):
    wrist = hand.landmark[0]
    index_mcp = hand.landmark[5]
    pinky_mcp = hand.landmark[17]
    dx = pinky_mcp.x - index_mcp.x
    dy = pinky_mcp.y - index_mcp.y
    return math.atan2(dy, dx)

def detect_gesture(hand, label, both_hands):
    gesture = None
    fingers = get_extended_fingers(hand)

    if label == "Left":
        if is_open_palm(hand):
            gesture = "MOVE_FORWARD"
        elif only_index(hand):
            gesture = "MOVE_LEFT"
        elif fingers[1] and not any([fingers[0], fingers[2], fingers[3]]):
            gesture = "MOVE_RIGHT"
        else:
            angle = get_wrist_angle(hand)
            if abs(angle) > 0.6:
                gesture = "MOVE_BACK"

    elif label == "Right":
        if is_fist(hand):
            gesture = "SHOOT"
        elif is_open_palm(hand):
            gesture = "JUMP"
        elif index_middle(hand):
            gesture = "AIM"
        elif thumb_pinky(hand):
            gesture = "RELOAD"
        else:
            # thumb gestures
            thumb_tip = hand.landmark[4]
            thumb_mcp = hand.landmark[2]
            angle = thumb_tip.y - thumb_mcp.y
            if angle < -0.05:
                gesture = "CROUCH_UP"
            elif angle > 0.05:
                gesture = "CROUCH_DOWN"

    if both_hands and is_fist(hand):
        gesture = "SPRINT"

    return gesture


# ==== Main Loop ====
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    current_gesture = None

    both_hands = results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(hand_landmarks, label, both_hands)
            if gesture:
                current_gesture = gesture
                send_command(gesture)
                cv2.putText(frame, f"{label}: {gesture}", (10, 40 if label == 'Left' else 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if current_gesture is None:
        send_command("STOP")

    cv2.putText(frame, "Gesture Control Active", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Gesture Control (Hands Only)", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
hands.close()
socket.close()
context.term()
cv2.destroyAllWindows()
print("Closed cleanly.")
