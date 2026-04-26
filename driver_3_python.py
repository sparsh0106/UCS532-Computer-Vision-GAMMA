import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance as dist

# ===============================
# PARAMETERS
# ===============================
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MAR_THRESH = 0.75

# important ratios to consider, EAR and MAR

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

#loading the detector!

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

# running the code on an image (if available)

def run_on_dataset(folder_path):

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

    folder_path = "/media/sparsh-sigma/CaptainSlow/Programming Stuff/Programming Stuff/Code _n_ Stuff/All Projects/All Projects/Computer Vision sem 6"   # keep or remove as needed
    

    for file in os.listdir(folder_path):

        if not file.lower().endswith(valid_ext):
            continue

        img_path = os.path.join(folder_path, file)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        print(f"\nProcessing Image: {file}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print("→ No face detected")
            cv2.putText(frame, "no face detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        for i, face in enumerate(faces):
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[LEFT_EYE]
            rightEye = shape[RIGHT_EYE]
            mouth = shape[MOUTH]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Draw landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0,255,0), -1)

            # Decision flags
            drowsy = ear < EYE_AR_THRESH
            yawning = mar > MAR_THRESH

            # Console output
            print(f"→ Face {i+1}:")
            print(f"   Drowsy: {'YES' if drowsy else 'NO'}")
            print(f"   Yawning: {'YES' if yawning else 'NO'}")

            # Overlay text
            if drowsy:
                cv2.putText(frame, "eyes closed (drowsy)", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if yawning:
                cv2.putText(frame, "yawning", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("Dataset Testing", frame)

        key = cv2.waitKey(500) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

# running the code on webcam!
def run_webcam():

    COUNTER = 0
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            cv2.putText(frame, "no face detected", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        for face in faces:
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[LEFT_EYE]
            rightEye = shape[RIGHT_EYE]
            mouth = shape[MOUTH]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0,255,0), -1)


            if ear < EYE_AR_THRESH:      # logic for drowsy
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "eyes closed (drowsy)", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                COUNTER = 0


            if mar > MAR_THRESH:         # logic for yawming
                cv2.putText(frame, "yawming", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)


        cv2.imshow("detection using webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    choice = str(input("A. wecbcam (enter 1)\nB. sample image (enter 2)\n: "))

    if choice == "1":
        run_webcam()

    elif choice == "2":
        folder = input("Enter dataset folder path: ")
        run_on_dataset(folder)

    else:
        print("Invalid choice")