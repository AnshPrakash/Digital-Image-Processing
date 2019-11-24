import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from utils import *

Nangles = 3
size = 3
sigma = 1
Levels = 3
kernels = gausianKernel(size,sigma,Nangles)

def getcdf(im):
  hist, bin_edges = np.histogram(im, bins=256,range=(0,255) ,density=True)
  cdf = np.zeros(bin_edges.shape)
  cdf[1:] = np.cumsum(hist*np.diff(bin_edges))
  return(cdf)

def inverse_transform_sampling(data, n_bins=256):
  hist, bin_edges = np.histogram(data, bins=n_bins,range=(0,255) ,density=True)
  cum_values = np.zeros(bin_edges.shape)
  cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
  inv_cdf = spi.interp1d(cum_values,bin_edges,fill_value="extrapolate")
  return inv_cdf

def matchHistogram(im1,im2):
  '''
    match im2 ->im1
  '''
  img = np.copy(im1)
  im1_cdf = getcdf(im1)
  im2_cdf = getcdf(im2)
  inv_im2_cdf = inverse_transform_sampling(im2)
  img = inv_im2_cdf(im1_cdf[im1])
  return(img)



def steerablePyramids(img,levels,x=[]):
  if levels <= 0 or img.shape[0]==1 or img.shape[1] == 1:
    x.append(img)
    return(x)
  else:
    l = [cv2.filter2D(img,-1,ker) for ker in kernels]
    x.append(l)
    steerablePyramids(cv2.pyrDown(img,(img.shape[1]//2,img.shape[0]//2)),levels-1,x)
    return(x)


def collapsePyramid(pyr):
  currImg = pyr[0]
  for lev in range(len(pyr) - 1):
    w = pyr[lev+1][0].shape[1]
    h = pyr[lev+1][0].shape[0]
    # img = cv2.resize(currImg,(w,h),interpolation = cv2.INTER_CUBIC)
    img = cv2.pyrUp(currImg,(w,h))
    # img = cv2.resize(currImg,(w,h),interpolation = cv2.INTER_AREA)
    sumIm = img
    for im in pyr[lev + 1]:
      sumIm = sumIm + im
    currImg = sumIm
  return(currImg)


def matchTexture(noise,texture):
  noise = matchHistogram(noise,texture)
  analysis_pyr = steerablePyramids(texture,Levels)
  iters = 1
  for _ in range(iters):
    synthesis_pyramid = steerablePyramids(noise,Levels)
    for lev in range(1,Levels+1):
      for i in range(len(synthesis_pyramid[lev])):
        synthesis_pyramid[lev][i] = matchHistogram(synthesis_pyramid[lev][i],analysis_pyr[lev][i])
    synthesis_pyramid[0] = matchHistogram(synthesis_pyramid[0],analysis_pyr[0])
    noise = collapsePyramid(synthesis_pyramid)
    noise = matchHistogram(noise,texture)
  return(noise)

  




texture = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
noise = (np.random.rand(texture.shape[0],texture.shape[1])*255).astype(np.uint8)

noiseOrg = np.copy(noise)

# exptexture = matchTexture(noise,texture)
exptexture = matchHistogram(noise,texture)

cv2.imshow("Noise",noiseOrg)
cv2.imshow("NewTexture",exptexture/255)
cv2.waitKey(0)
cv2.destroyAllWindows()



  
