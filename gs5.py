import cv2
import mediapipe as mp
import zmq
import math

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
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Changed to detect both hands
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
    elif fingers[0] and not any(fingers[i] for i in [1,2,3,4]):
        return "JUMP"  # Thumb up
    else:
        return "NONE"

def detect_movement(left_hand, right_hand):
    """Detect movement based on left hand state and index fingers."""
    movement = "STOP"
    
    # Check for left hand movements first
    if left_hand:
        left_fingers = get_finger_states(left_hand.landmark)
        left_total_extended = left_fingers.count(True)
        
        # Debug print for left hand
        #print(f"Left fingers: {left_fingers}, Total extended: {left_total_extended}")
        
        # Left hand movements
        if left_total_extended == 3:  # Open palm
            movement = "MOVE_FORWARD"
            #print("MOVE_FORWARD detected")
        elif left_fingers[1] and not any(left_fingers[i] for i in [2,3,4]):  # Index only
            movement = "MOVE_LEFT"
            #print("MOVE_LEFT detected")
        elif left_total_extended <= 1:  # Closed palm/fist
            movement = "MOVE_BACKWARD"
            #print("MOVE_BACKWARD detected")
        
    
    # Check for right hand index finger (move right) - only if no left movement detected
    if right_hand:
        right_fingers = get_finger_states(right_hand.landmark)
        
        # Debug print for right hand
        #print(f"Right fingers: {right_fingers}")
        
        if right_fingers[1] and not any(right_fingers[i] for i in [0,2,3,4]):  # Index only
            movement = "MOVE_RIGHT"
            #print("MOVE_RIGHT detected")
    
    return movement
# ======================
# Look Control (Hand Delta)
# ======================
prev_face_pos = None  # global variable, just like prev_right_hand_pos

def get_face_look_delta(face_landmarks):
    """Compute smooth camera look delta from face (nose) movement."""
    global prev_face_pos

    if not face_landmarks:
        prev_face_pos = None
        return (0.0, 0.0)

    # Nose tip (landmark index 1 is usually the nose tip)
    x = face_landmarks[1].x
    y = face_landmarks[1].y

    dx, dy = 0.0, 0.0
    if prev_face_pos:
        dx = x - prev_face_pos[0]
        dy = prev_face_pos[1] - y

    prev_face_pos = (x, y)

    # Dead zone filter to reduce jitter
    if abs(dx) < 0.002: dx = 0.0
    if abs(dy) < 0.002: dy = 0.0

    # Sensitivity and smoothing (same logic as your hand version)
    sensitivity = 5.0     # increase for faster camera rotation
    smooth_factor = 0.75  # higher = smoother, but slower response

    dx *= sensitivity * (1 - smooth_factor)
    dy *= sensitivity * (1 - smooth_factor)

    # Clamp to avoid large jumps
    dx = max(-0.5, min(0.5, dx))
    dy = max(-0.5, min(0.5, dy))

    return (dx, dy)
# ======================
# Main Loop
# ======================
cap = cv2.VideoCapture(0)
print("Gesture detection started. Showing webcam...")
look_enabled = False  # start disabled by default
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("⚠ Skipping empty frame")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    face_results = face_mesh.process(rgb)
    gesture = "NONE"
    movement = "NONE"
    look_dx, look_dy = 0.0, 0.0
    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks:
        # Identify left and right hands
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                            results.multi_handedness):
            hand_label = handedness.classification[0].label
            
            if hand_label == "Left":
                left_hand = hand_landmarks
                # Draw left hand in blue
                mp_drawing.draw_landmarks(frame, hand_landmarks, 
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
                
            else:  # Right hand
                right_hand = hand_landmarks
                # Draw right hand in green
                mp_drawing.draw_landmarks(frame, hand_landmarks, 
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
        
        # Detect gesture from right hand
        if right_hand:
            gesture = recognize_gesture(right_hand.landmark)
        
        # Gesture-based toggle for look control
            
            if gesture == "STOP":      # Open palm → start look
                look_enabled = True
            elif gesture == "SHOOT":   # Fist → stop look
                look_enabled = False
        # Detect movement from both hands
        movement = detect_movement(left_hand, right_hand)

         # Camera Look Delta
        # Detect face for look
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark
        if look_enabled and right_hand:
            look_dx, look_dy = get_face_look_delta(face_landmarks)
        else:
            look_dx, look_dy = 0.0, 0.0

        # Draw nose position
        h, w, _ = frame.shape
        nose_x, nose_y = int(face_landmarks[1].x * w), int(face_landmarks[1].y * h)
        cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 255), -1)
        cv2.putText(frame, "Face Tracking Active", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)



        # if gesture == "STOP":
        #     movement = "NONE"
    # Send combined command to Unity (format: "GESTURE|MOVEMENT")
    #command = f"{gesture}|{movement}"
    if movement == "STOP":
        movement = "NONE"
    command = f"{gesture}|{movement}|{look_dx:.3f},{look_dy:.3f}"
    socket.send_string(command)
    
    # Display information on frame
    cv2.putText(frame, f"Right Hand (Gesture): {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Left Hand (Movement): {movement}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Command: {command}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add finger state debug info
    if left_hand:
        left_fingers = get_finger_states(left_hand.landmark)
        cv2.putText(frame, f"Left Fingers: {[1 if f else 0 for f in left_fingers]}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if right_hand:
        right_fingers = get_finger_states(right_hand.landmark)
        cv2.putText(frame, f"Right Fingers: {[1 if f else 0 for f in right_fingers]}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("Gesture & Movement Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()