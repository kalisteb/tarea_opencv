import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

# creando una cascada de rostros
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

# establece la fuente de video en la cámara web predeterminada
video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(6)
        pass

    # Capturamos el video
    ret, frame = video_capture.read()

    # lo convertimos a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecta la cara en la imagen
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(25, 25)
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Dibuja un rectángulo alrededor del rostro
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Muestra el marco resultante
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Muestra el marco resultante
    cv2.imshow('Video', frame)

# Cuando está todo listo suelta la captura
video_capture.release()
cv2.destroyAllWindows()