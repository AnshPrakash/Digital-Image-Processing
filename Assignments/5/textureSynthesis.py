import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import *

Nangles = 3
size = 3
sigma = 1
Levels = 3

def getcdf(im):
  cdf = [0]*256
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      cdf[im[i][j]] += 1
  for i in range(1,len(cdf)):
    cdf[i] = cdf[i]+ cdf[i-1]
  cdf = np.array(cdf)/im.size
  return(cdf)

def matchHistogram(im1,im2):
  img = np.copy(im1)/1.0
  im1_cdf = getcdf(im1)
  im2_cdf = getcdf(im2)
  # inv_im2_cdf = np.array(im2_cdf)*255
  for i in range(im1.shape[0]):
    for j in range(im1.shape[1]):
      # img[i][j] = inv_im2_cdf[im1_cdf[img[i][j]]]
      img[i][j] = im1_cdf[im1[i][j]]
  return(img)


kernels = gausianKernel(size,sigma,Nangles)

def steerablePyramids(img,levels,x=[]):
  if levels <= 0 or img.shape[0]==1 or img.shape[1] == 1:
    x.append(img)
    return(x)
  else:
    l = [cv2.filter2D(img,-1,ker) for ker in kernels]
    x.append(l)
    steerablePyramids(cv2.resize(img,(img.shape[1]//2,img.shape[0]//2), interpolation = cv2.INTER_AREA),levels-1,x)
    return(x)







img1 = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)


pyr = steerablePyramids(img1,Levels)
pyr.reverse()

print(len(pyr))
displayPyramid(pyr,Levels)

# im = matchHistogram(img1,img2)
# cv2.imshow("image",img1)
# cv2.imshow("histoMatch",im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



  
