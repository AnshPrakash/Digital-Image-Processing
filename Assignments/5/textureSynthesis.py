import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import *


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


def gausianKernel(size,sigma,Nangles):
  angles = [ i*(3.14/Nangles) for i in range(Nangles+1)]
  cols = [np.array(list(range(size))) - (size-1)/2.0 for _ in range(size)]
  cols = np.array(cols)
  rows = np.copy(cols).T
  g = -np.exp(-(cols*cols + rows*rows)/2*sigma*sigma)/(2*3.14*(sigma**4))
  gx = rows*g
  gy = cols*g
  ga = [np.cos(alp)*gx + np.sin(alp)*gy for alp in angles]
  return(ga)






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

Nangles = 3
size = 3
sigma = 1
kernels = gausianKernel(size,sigma,Nangles)

Levels = 3
pyr = steerablePyramids(img1,Levels)
pyr.reverse()

print(len(pyr))
displayPyramid(pyr,Levels)

# im = matchHistogram(img1,img2)
# cv2.imshow("image",img1)
# cv2.imshow("histoMatch",im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



  
