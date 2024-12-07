import cv2
import mediapipe as mp
import pygame
import math

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

try:
    pygame.mixer.init()
    pygame.mixer.music.load('alarm.wav')  # Ensure 'alarm.wav' path is correct
except Exception as e:
    print("Error initializing pygame or loading sound file:", e)
    pygame_error = True
else:
    pygame_error = False
  # Add the path to your alarm sound file

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Eye and mouth landmarks
LEFT_EYE = [33, 133, 160, 159, 158, 157, 154, 153, 144]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 381, 380, 374]
MOUTH = [13, 14, 78, 308]  # Points for the mouth

# Indices for calculating head rotation (nose bridge)
NOSE_BRIDGE = [1, 168]

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

# Function to calculate head yaw angle (sideways head rotation)
def calculate_yaw(nose_bridge, landmarks):
    nose_start = landmarks[nose_bridge[0]]  # Top of the nose bridge
    nose_end = landmarks[nose_bridge[1]]  # Bottom of the nose bridge
    
    # Calculate the yaw angle (horizontal deviation)
    yaw = math.degrees(math.atan2(nose_end.x - nose_start.x, nose_end.y - nose_start.y))
    return yaw

# Drowsiness detection thresholds
EAR_THRESHOLD = 0.25
CLOSED_EYE_FRAMES = 30  # Number of frames eyes should be closed to detect drowsiness
MAR_THRESHOLD = 1.0     # Threshold for yawning

closed_eyes_frame_count = 0
no_face_frame_count = 0
alarm_playing = False

# Maximum allowed frames without face detection before triggering alarm
NO_FACE_THRESHOLD = 20  # Adjust based on how strict you want the face detection to be

# Yaw angle thresholds
YAW_LEFT_THRESHOLD = -20  # Yaw angle for leftward turn
YAW_RIGHT_THRESHOLD = 20  # Yaw angle for rightward turn

# Start the Face Mesh model
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image for face landmarks
        face_result = face_mesh.process(rgb_frame)

        # Convert the image back to BGR for display
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Check if face landmarks are detected
        if face_result.multi_face_landmarks:
            no_face_frame_count = 0  # Reset count when face is detected
            face_landmarks = face_result.multi_face_landmarks[0]

            # Calculate EAR for both eyes
            left_ear = calculate_ear(LEFT_EYE, face_landmarks.landmark)
            right_ear = calculate_ear(RIGHT_EYE, face_landmarks.landmark)
            avg_ear = (left_ear + right_ear) / 2

            # Calculate MAR
            mar = calculate_mar(MOUTH, face_landmarks.landmark)

            # Calculate yaw angle (head rotation)
            yaw = calculate_yaw(NOSE_BRIDGE, face_landmarks.landmark)
            print(f"Yaw Angle: {yaw}")  # Debug: Print yaw angle

            # Check for sideways head turn based on yaw angle
            if yaw < YAW_LEFT_THRESHOLD:
                cv2.putText(frame, "HEAD TURNED LEFT!", (100, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

                # Play alarm if head is turned too much left
                if not alarm_playing:
                    pygame.mixer.music.play(-1)
                    alarm_playing = True
            elif yaw > YAW_RIGHT_THRESHOLD:
                cv2.putText(frame, "HEAD TURNED RIGHT!", (100, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 4)

                # Play alarm if head is turned too much right
                if not alarm_playing:
                    pygame.mixer.music.play(-1)
                    alarm_playing = True

            # Check for drowsiness based on EAR
            if avg_ear < EAR_THRESHOLD:
                closed_eyes_frame_count += 1
            else:
                closed_eyes_frame_count = 0

            # Check for yawning based on MAR
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWNING DETECTED!", (100, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)

                # Play alarm if yawning is detected
                if not alarm_playing:
                    pygame.mixer.music.play(-1)
                    alarm_playing = True

            # If eyes have been closed for a sufficient number of frames, trigger drowsiness alert
            if closed_eyes_frame_count >= CLOSED_EYE_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                # Play alarm if drowsiness is detected
                if not alarm_playing:
                    pygame.mixer.music.play(-1)  # Play the alarm continuously
                    alarm_playing = True
            else:
                # Stop the alarm if neither drowsiness nor yawning or head turn is detected
                if alarm_playing and closed_eyes_frame_count == 0 and mar < MAR_THRESHOLD and (YAW_LEFT_THRESHOLD < yaw < YAW_RIGHT_THRESHOLD):
                    pygame.mixer.music.stop()
                    alarm_playing = False

        else:
            # Increment the count if the face is not detected
            no_face_frame_count += 1

            # Trigger alarm if no face is detected for a certain number of frames
            if no_face_frame_count >= NO_FACE_THRESHOLD:
                cv2.putText(frame, "NO FACE DETECTED", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                if not alarm_playing:
                    pygame.mixer.music.play(-1)  # Play alarm continuously when face is lost
                    alarm_playing = True

        # Display the frame
        cv2.imshow("Drowsiness and Sideways Head Detection", frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
