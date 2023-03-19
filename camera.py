import cv2
import numpy as np

img1 = cv2.imread('./images/photo3.jpg')

img_32bit = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)

cv2.imwrite('32_bit_image3.png', img_32bit)
