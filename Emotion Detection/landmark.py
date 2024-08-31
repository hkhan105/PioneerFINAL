import cv2
import mediapipe as mp
import time

# Initialize face mesh detector
cap = cv2.VideoCapture(0)
previous_time = 0

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=3, circle_radius=3)

if not cap.isOpened():
    print("Can't open camera")
    exit(0)

def emotion_rules(landmarks):
    left_eye_open = landmarks[145].y - landmarks[159].y
    right_eye_open = landmarks[374].y - landmarks[386].y
    mouth_open = landmarks[13].y - landmarks[14].y
    left_eyebrow_y = landmarks[65].y - landmarks[159].y
    right_eyebrow_y = landmarks[295].y - landmarks[386].y
    mouth_width = landmarks[291].x - landmarks[61].x
    upper_lip_y = landmarks[0].y - landmarks[13].y

    if mouth_open > 0.05 and left_eye_open > 0.04 and right_eye_open > 0.04 and left_eyebrow_y > 0.02 and right_eyebrow_y > 0.02:
        return "Surprise"
    elif mouth_open < 0.02 and left_eye_open < 0.02 and right_eye_open < 0.02 and left_eyebrow_y < -0.01 and right_eyebrow_y < -0.01 and mouth_width < 0.05:
        return "Anger"
    elif mouth_open < 0.03 < right_eye_open and left_eye_open > 0.03 and left_eyebrow_y > 0.02 and right_eyebrow_y > 0.02 and mouth_width < 0.04:
        return "Sadness"
    elif mouth_open > 0.05 and left_eye_open < 0.03 and right_eye_open < 0.03 and mouth_width > 0.05 and upper_lip_y < 0.02:
        return "Joy"
    elif left_eye_open > 0.05 and right_eye_open > 0.05 and left_eyebrow_y > 0.03 and right_eyebrow_y > 0.03 and mouth_open > 0.02:
        return "Fear"
    elif upper_lip_y < -0.01 and left_eye_open < 0.02 and right_eye_open < 0.02:
        return "Disgust"
    elif (landmarks[61].y - landmarks[291].y) > 0.02:
        return "Contempt"

emotion_colors = {
    "Joy": (215, 245, 66),
    "Surprise": (66, 176, 245),
    "Sadness": (84, 24, 14),
    "Anger": (40, 14, 84),
    "Fear": (84, 14, 77),
    "Disgust": (20, 3, 19),
    "Contempt": (230, 225, 229)
}

while True:
    success, img = cap.read()
    if success:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, draw_spec, draw_spec)
                landmarks = face_landmarks.landmark
                emotion = classify_emotion(landmarks)
                color = emotion_colors.get(emotion, (255, 255, 255))
                cv2.putText(img, emotion, (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
                cv2.rectangle(img, (0, 0), (50, 50), color, thickness=-1)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Emotion Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Can't find emotion")
        break

cap.release()
cv2.destroyAllWindows()
