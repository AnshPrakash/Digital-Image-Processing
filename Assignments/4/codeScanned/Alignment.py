import numpy as np
import imutils
from matplotlib import pyplot as plt
import sys,os
import cv2


def findReflection(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  return(mask)


img = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (900,900), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blurred = cv2.medianBlur(gray,5)
sigI,sigS = 5,5
kbsize = -1
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)

edges = cv2.Canny(blurred,60,130)

rho = 1
theta = 3.14/180.0
threshold = 150
min_theta = 1.4
max_theta = 1.8
lines = cv2.HoughLines(	edges, rho, theta, threshold, min_theta, max_theta	)
thetas = []
for line in lines:
  for rho,theta in line:
      thetas.append(theta)
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))
      # cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

angle = 90 - (np.median(thetas))*(180/3.14)

rotated = imutils.rotate_bound(img, angle)

# mask = findReflection(blurred)
cv2.imshow("Edges",edges)
cv2.imshow("Original",img)
cv2.imshow("Rotated",rotated)


# cv2.imwrite("Original.jpg",img)
# cv2.imwrite("Rotated.jpg",rotated)


cv2.waitKey(0)
cv2.destroyAllWindows()

