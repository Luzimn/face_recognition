import threading

import cv2

from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


counter = 0

face_math = False

reference_img = cv2.imread("reference.jpg")


def check_face(frame):
    global face_math
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_math = True
        else:
            face_math = False



    except ValueError:
        face_math = False


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()

            except ValueError:
                pass
        counter += 1

        if face_math:
            cv2.putText(frame, "aprove!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "not aprove!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)


        cv2.imshow("video", frame)


    Key = cv2.waitKey(1)
    if Key == ord("q"):
        break

cap.release() 
cv2.destroyAllWindows()