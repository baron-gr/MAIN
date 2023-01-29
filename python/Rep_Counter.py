import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

counter = 0
counter_left = 0
counter_right = 0
position = None
position_left = None
position_right = None

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs((radians* 180.0) / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

with mp_pose.Pose(min_detection_confidence = 0.9, min_tracking_confidence = 0.9) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        results = pose.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
            cv2.putText(image, str(angle_left), tuple(np.multiply(elbow_left, [1920, 1080]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            if angle_left > 160:
                position_left = "down"
            if angle_left < 30 and position_left == 'down':
                position_left = "up"
                counter_left += 1
        except:
            pass
    
        try:
            landmarks = results.pose_landmarks.landmark

            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)
            cv2.putText(image, str(angle_right), tuple(np.multiply(elbow_right, [1920, 1080]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            if angle_right > 160:
                position_right = "down"
            if angle_right < 30 and position_right == 'down':
                position_right = "up"
                counter_right += 1
        except:
            pass

        try:
            landmarks = results.pose_landmarks.landmark

            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            angle = calculate_angle(hip, knee, ankle)
            cv2.putText(image, str(angle), tuple(np.multiply(knee, [1920, 1080]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            if angle > 170:
                position = "up"
            if angle < 100 and position == 'up':
                position = "down"
                counter += 1
        except:
            pass

        cv2.rectangle(image, (0,0), (300,90), (245,117,16), -1)
        cv2.rectangle(image, (1920,0), (1620,90), (245,117,16), -1)
        cv2.rectangle(image, (0,1080), (300,990), (245,117,16), -1)

        cv2.putText(image, 'Reps', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_left), (25,75), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Position', (140,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, position_left, (140,70), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'Reps', (1640,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter_right), (1645,75), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Position', (1760,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, position_right, (1760,70), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'Reps', (20,1010), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (25,1065), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Position', (140,1010), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, position, (140,1060), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (255,255,255), 2, cv2.LINE_AA)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255,0,0), thickness=4, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=4, circle_radius=4))               
        
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()