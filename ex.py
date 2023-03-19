import cv2
import numpy as np
import argparse

# yüz algılama sınıflandırıcısı
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv2.CascadeClassifier()


# tanınacak kişilerin resimleri
img1 = cv2.imread('images/photo1.jpg')
img2 = cv2.imread('images/photo2.jpg')
img3 = cv2.imread('images/photo3.jpg')

# kişilerin isimleri
names = ['candas', 'flavio', 'fateme']

def detectAndDisplay(img1,img2,img3):

    face_cascade.load(face_cascade_name)
    frame_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    frame_gray1 = cv2.equalizeHist(frame_gray1)
    frame_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    frame_gray2 = cv2.equalizeHist(frame_gray2)
    frame_gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    frame_gray3 = cv2.equalizeHist(frame_gray3)
    #-- Detect faces
    faces1 = face_cascade.detectMultiScale(frame_gray1)
    faces2 = face_cascade.detectMultiScale(frame_gray2)
    faces3 = face_cascade.detectMultiScale(frame_gray3)
    for (x,y,w,h) in faces1:
        # center = (x + w//2, y + h//2)
        img1 = cv2.rectangle(img1, (x,y), (x+w,y+h), (0,255,0) ,2)
        faceROI = cv2.putText(img1, names[0], (x,y+h+20), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,0))
    
    for (x,y,w,h) in faces2:
        # center = (x + w//2, y + h//2)
        img2 = cv2.rectangle(img2, (x,y), (x+w,y+h), (0,255,0) ,2)
        faceROI = cv2.putText(img2, names[1], (x,y+h+20), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,0))

    for (x,y,w,h) in faces3:
        # center = (x + w//2, y + h//2)
        img3 = cv2.rectangle(img3, (x,y), (x+w,y+h), (0,255,0) ,2)
        faceROI = cv2.putText(img3, names[2], (x,y+h+20), cv2.FONT_HERSHEY_DUPLEX, .5, (0,255,0))
        #-- In each face, detect eyes
        
    cv2.imshow('candas',img1)
    cv2.imshow('flavio',img2)
    cv2.imshow('fateme',img3)

camera_device = args.camera
#-- 2. Read the video stream
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(img1,img2,img3)
    if cv2.waitKey(20) & 0xFF == ord('x'):
        break

