import numpy as np
import cv2
import sys
import imutils

# mser = cv2.MSER_create(_delta = 1,_max_area=1500,_min_area = 90,_max_variation = 0.03,_min_diversity = 10,_edge_blur_size=1)
mser = cv2.MSER_create(_delta = 2,_max_area=1900,_min_area = 30,_max_variation = 0.2)


def binarize(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,151,-5)
  return(mask)


def visulalizeMarker(markers):
  marks = np.copy(markers)
  marks = marks.astype(np.uint8)
  marks = cv2.applyColorMap(marks, cv2.COLORMAP_JET)
  return(marks)




img = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
print(img.shape)
# img = cv2.resize(img, (900,900), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sigI,sigS = 5,5
kbsize = -1
blurred = cv2.bilateralFilter(gray,kbsize,sigI,sigS)
blurred = cv2.bilateralFilter(gray,kbsize,sigI,sigS)
edges = cv2.Canny(blurred,60,130)

kernel = np.ones((3,3),np.uint8)
remover = binarize(blurred)
remover = cv2.morphologyEx(remover, cv2.MORPH_OPEN, kernel,iterations = 5)
remover =  cv2.dilate(remover,kernel,iterations=6)
edges = edges*(remover/255)
edges = edges.astype(np.uint8)
# remover =  cv2.erode(remover,kernel,iterations=3)

vis = img.copy()
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
white_bg = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)*255

# mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))

regions, boundingBoxes = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]


# charHulls = []
for i,hull in enumerate(hulls):
  area = cv2.contourArea(hull)
  if area < 1050:
    # charHulls.append(hull)
    x, y, w, h = boundingBoxes[i]
    roi = blurred[y:y + h, x:x + w]
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # mask(boundingBoxes[i]).setTo(255)
    white_bg[y:y+h, x:x+w] = roi
    cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), 1)


intersect = ((1 - white_bg/255)*(edges/255)*255)
intersect = intersect.astype('uint8')
ret2,intersect = cv2.threshold(intersect,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(intersect, cv2.MORPH_CLOSE, kernel,iterations = 1)


_,contours,_ = cv2.findContours(closing,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 


epsilon = 0.55
error_peri = 1

conts_im = cv2.findContours(closing,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
# conts_im = cv2.findContours(closing,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
# print(conts_im)
cnts_im = imutils.grab_contours(conts_im)
mask_im = np.ones(img.shape[:2], dtype="uint8") 

FinalResult = closing.copy()

closing = cv2.cvtColor(closing,cv2.COLOR_GRAY2BGR)
for i in range(0, len(contours)):
  cnt = contours[i]
  area = cv2.contourArea(cnt)
  perimeter = cv2.arcLength(cnt,True)
  if area > 150:
    rect = cv2.minAreaRect(cnt)
    (x, y), (width, height), angle = rect
    aspect_ratio = min(width, height) / max(width, height)
    peri_area = perimeter/area
    if ((1 - aspect_ratio) < epsilon) or not (peri_area < 1):
      continue
    print(1 -aspect_ratio)
    print("p",peri_area)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(mask_im, [cnts_im[i]], -1, 0, -1)
    cv2.drawContours(closing,[box],0,(0,0,255),2)
    # cv2.rectangle(closing,(x,y),(x+w,y+h),(255,0,0),2)



# cv2.drawContours(closing, contours, -1, (0, 255, 0), 3)


FinalResult = FinalResult*mask_im
# FinalResult = cv2.morphologyEx(FinalResult, cv2.MORPH_CLOSE, kernel,iterations = 1)
ret, markers = cv2.connectedComponents(FinalResult)
visualMarks = visulalizeMarker(markers)

cv2.imwrite('Char.jpg',vis)
cv2.imwrite('Mask.jpg',mask)
cv2.imwrite('Extract.jpg',white_bg)
cv2.imwrite('Intersection.jpg',intersect)
cv2.imwrite('Morpho.jpg',closing)
cv2.imwrite('Visualize.jpg',visualMarks)
cv2.imwrite('Detect_FalsePositive.jpg',mask_im*255)
cv2.imwrite('FinalResult.jpg',FinalResult)
cv2.imwrite('Remover.jpg',remover)



