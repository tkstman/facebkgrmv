import cv2
import os
import numpy as np
import facedetection as fr 

test_img=cv2.imread('C:/Users/tstone/Documents/Python/testimgs/0000000.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)

print("faces_detected:",faces_detected)

for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows