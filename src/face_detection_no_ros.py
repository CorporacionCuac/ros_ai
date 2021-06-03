#!/usr/bin/env python3

import cv2

def detectAndDisplay(frame, face_detector):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #-- Detect faces
    faces = face_detector.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow('Capture - Face detection', frame)


def main():
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(frame, face_detector)
        if cv2.waitKey(10) == 27:
            break

if __name__ == '__main__':
    main()
