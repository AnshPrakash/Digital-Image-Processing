import numpy as np
import pywt
from matplotlib import pyplot as plt
import sys,os
import cv2


def build_filters(lambdda = 10.0,sigma=9.0,gamma = 5.0,psi = 0.0):
  filters = []
  ksize = 11
  for theta in np.arange(0, np.pi, np.pi / 16.0):
    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambdda, gamma, psi, ktype = cv2.CV_32F)
    kern /= 1.5*kern.sum()
    filters.append(kern)
  return filters

def processGabor(img, filters):
  accum = np.zeros_like(img)
  for kern in filters:
    fimg = cv2.filter2D(img, cv2.CV_8U, kern)
    np.maximum(accum, fimg, accum)
  return accum


def binarize(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
  return(mask)

def visulalizeMarker(markers):
  marks = np.copy(markers)
  marks = marks.astype(np.uint8)
  marks = cv2.applyColorMap(marks, cv2.COLORMAP_JET)
  return(marks)




img = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (900,900), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



# img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)

blurred = cv2.medianBlur(gray,5)
sigI,sigS = 5,5
kbsize = -1
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
# filters = build_filters()
# response = processGabor(blurred,filters)

kernel = np.ones((3,3),np.uint8)
sure_fg = binarize(blurred)
sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel,iterations = 2)
# sure_fg =  cv2.dilate(sure_fg,kernel,iterations=3)
sure_fg =  cv2.erode(sure_fg,kernel,iterations=3)
sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel,iterations = 1)



# sure_fg = 255 - sure_fg
# sure_fg = sure_fg.astype(np.int32)
ret, markers = cv2.connectedComponents(sure_fg)
visualMarks = visulalizeMarker(markers)
print(ret)

# print(img.dtype)
# print(markers.dtype)
# print(sure_fg.dtype)
# markers = markers + 1
seg = cv2.watershed(img,markers)
img[seg == -1] = [25,223,212]
print(seg.dtype)
print(seg.shape)


seg = visulalizeMarker(seg)

cv2.imshow("Original",img)
cv2.imshow("sure_fg",sure_fg)
# cv2.imshow("markers",markers/255.0)
cv2.imshow("visualize Markers",visualMarks)
cv2.imshow("Segmentations",seg)



# cv2.imshow("Segmentes",seg)



# cv2.imshow("Gabor",response)
cv2.waitKey(0)
cv2.destroyAllWindows()

