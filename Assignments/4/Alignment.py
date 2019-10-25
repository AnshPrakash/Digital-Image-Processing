import numpy as np
import pywt
from matplotlib import pyplot as plt
import sys,os
import cv2


def findReflection(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  return(mask)


img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
# img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

blurred = cv2.medianBlur(img,5)
sigI,sigS = 5,5
kbsize = -1
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
# img = img/255.0

mask = findReflection(blurred)
cv2.imshow("Original",img)
cv2.imshow("Reflection Detected",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

