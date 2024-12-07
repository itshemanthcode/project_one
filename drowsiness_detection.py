import cv2
import mediapipe as mp
import pygame
import math

# Initialize Mediapipe Face Mesh and Pose
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize pygame for the alarm sound
pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')  # Add a path to your alarm sound file

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Eye and mouth landmarks
LEFT_EYE = [33, 133, 160, 159, 158, 157, 154, 153, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 381, 380, 374]
MOUTH = [13, 14, 78, 308]  # Points for the mouth
NOSE_TIP = [1]  # Nose tip for forehead reference
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Additional landmarks for top of the head (forehead) and chin
TOP_HEAD = [10]  # You can choose a landmark that represents the forehead
BOTTOM_HEAD = [152]  # Chin landmark

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye_points, landmarks):
    left = landmarks[eye_points[0]]
    right = landmarks[eye_points[3]]
    top_left = landmarks[eye_points[1]]
    bottom_left = landmarks[eye_points[2]]
    top_right = landmarks[eye_points[5]]
    bottom_right = landmarks[eye_points[4]]

    # EAR calculation
    ear = (abs(top_left.y - bottom_left.y) + abs(top_right.y - bottom_right.y)) / (2 * abs(left.x - right.x))
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(mouth_points, landmarks):
    top_lip = landmarks[mouth_points[0]]
    bottom_lip = landmarks[mouth_points[1]]
    left_corner = landmarks[mouth_points[2]]
    right_corner = landmarks[mouth_points[3]]

    # MAR calculation
    mar = abs(top_lip.y - bottom_lip.y) / abs(left_corner.x - right_corner.x)
    return mar

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(ang)

# Drowsiness detection thresholds
EAR_THRESHOLD = 0.25
CLOSED_EYE_FRAMES = 30  # Number of frames eyes should be closed to detect drowsiness
MAR_THRESHOLD = 1.0     # Threshold for yawning
ALARM_ANGLE_THRESHOLD = 50  # Angle threshold for alarm

closed_eyes_frame_count = 0
alarm_playing = False

# Start the Face Mesh and Pose models
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh, \
        mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image for face and pose landmarks
        face_result = face_mesh.process(rgb_frame)
        pose_result = pose.process(rgb_frame)

        # Convert the image back to BGR for display
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Get face and pose landmarks
        if face_result.multi_face_landmarks and pose_result.pose_landmarks:
            face_landmarks = face_result.multi_face_landmarks[0]
            pose_landmarks = pose_result.pose_landmarks

            # Extract relevant face and shoulder points
            nose_tip = face_landmarks.landmark[NOSE_TIP[0]]
            left_shoulder = pose_landmarks.landmark[LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[RIGHT_SHOULDER]

            # Convert normalized coordinates to pixel values
            height, width, _ = frame.shape
            nose_tip_coords = (int(nose_tip.x * width), int(nose_tip.y * height))
            left_shoulder_coords = (int(left_shoulder.x * width), int(left_shoulder.y * height))
            right_shoulder_coords = (int(right_shoulder.x * width), int(right_shoulder.y * height))

            # Draw landmarks for visualization
            cv2.circle(frame, nose_tip_coords, 5, (0, 255, 0), -1)
            cv2.circle(frame, left_shoulder_coords, 5, (255, 0, 0), -1)
            cv2.circle(frame, right_shoulder_coords, 5, (255, 0, 0), -1)

            # Check for drowsiness based on EAR
            left_ear = calculate_ear(LEFT_EYE, face_landmarks.landmark)
            right_ear = calculate_ear(RIGHT_EYE, face_landmarks.landmark)
            avg_ear = (left_ear + right_ear) / 2

            # Calculate MAR
            mar = calculate_mar(MOUTH, face_landmarks.landmark)

            # Check for drowsiness based on EAR
            if avg_ear < EAR_THRESHOLD:
                closed_eyes_frame_count += 1
            else:
                closed_eyes_frame_count = 0

            # Determine the angle between the nose tip and the shoulders
            shoulder_angle_left = calculate_angle(nose_tip_coords, left_shoulder_coords, right_shoulder_coords)
            shoulder_angle_right = calculate_angle(nose_tip_coords, right_shoulder_coords, left_shoulder_coords)

            # Display the angles on the frame
            cv2.putText(frame, f"Left Shoulder Angle: {int(shoulder_angle_left)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Shoulder Angle: {int(shoulder_angle_right)}", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Check for yawning based on MAR
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED!", (100, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

            # Determine if the alarm should be played based on conditions
            if closed_eyes_frame_count >= CLOSED_EYE_FRAMES or mar > MAR_THRESHOLD:
                # Check if the angle between the nose tip and shoulders is less than the threshold
                if shoulder_angle_left < ALARM_ANGLE_THRESHOLD or shoulder_angle_right < ALARM_ANGLE_THRESHOLD:
                    if not alarm_playing:
                        pygame.mixer.music.play(-1)  # Play the alarm continuously
                        alarm_playing = True
                else:
                    # Stop the alarm if the angle is proper
                    if alarm_playing:
                        pygame.mixer.music.stop()
                        alarm_playing = False
            else:
                # Stop the alarm if eyes are not closed and yawning is not detected
                if alarm_playing:
                    pygame.mixer.music.stop()
                    alarm_playing = False

        # Show the video feed
        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
