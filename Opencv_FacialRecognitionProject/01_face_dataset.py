import cv2
import os

faceCascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_id = input("\nEnter user id end press <return> ==> ")
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

count = 0
while True:
    ret, img = cap.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + "." +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow("image", img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif count >= 1000:
        break

print("\n [INFO] Exiting program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()