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


'''
 All images are floating of floatiing point data type
'''
def showhistogram(im):
  minElement = np.amin(im)
  maxElement = np.amax(im)
  nbins = 256
  hist, bin_edges = np.histogram(im.ravel(), bins = nbins,range=(minElement,maxElement) ,density=True)
  cdf = np.zeros(bin_edges.shape)
  cdf[1:] = np.cumsum(hist*np.diff(bin_edges))
  plt.bar(bin_edges[:-1],cdf[1:], width=(maxElement- minElement)/nbins)
  plt.show()

def checkTypes(pyr,Levels):
  x = [pyr[0].dtype]
  for lev in range(1,Levels+1):
    l = []
    for i in range(len(pyr[lev])):
      l.append(pyr[lev][i].dtype)
    x.append(l)
  print(x)

def checkShapes(pyr,Levels):
  x = [pyr[0].shape]
  for lev in range(1,Levels+1):
    l = []
    for i in range(len(pyr[lev])):
      l.append(pyr[lev][i].shape)
    x.append(l)
  print(x) 

def getcdf(im):
  minElement = np.amin(im)
  maxElement = np.amax(im)
  nbins = 256
  hist, bin_edges = np.histogram(im.ravel(), bins = nbins,range=(minElement,maxElement) ,density=True)
  cdf = np.zeros(bin_edges.shape)
  cdf[1:] = np.cumsum(hist*np.diff(bin_edges))
  return(cdf,bin_edges)

def inverse_transform_sampling(im):
  minElement = np.amin(im)
  maxElement = np.amax(im)
  nbins = 256
  hist, bin_edges = np.histogram(im.ravel(), bins = nbins,range=(minElement,maxElement) ,density=True)
  cdf = np.zeros(bin_edges.shape)
  cdf[1:] = np.cumsum(hist*np.diff(bin_edges))
  inv_cdf = spi.interp1d(cdf,bin_edges,fill_value="extrapolate")
  return inv_cdf

def histogramEqualization(img):
  '''
    Required for negative value
  '''
  im = (img*255).astype(np.int)
  mini = np.amin(im)
  im = im - mini
  minElement = 0
  maxElement = np.amax(im)
  nbins = 256
  hist, bin_edges = np.histogram(im.ravel(), bins = nbins,range=(minElement,maxElement) ,density=True)
  cdf = np.zeros(bin_edges.shape)
  cdf[1:] = np.cumsum(hist*np.diff(bin_edges))
  # print(len(cdf))
  im = cdf[im]
  im = im + mini
  return(im/255.0)
  




def matchHistogram(im1,im2):
  '''
    match im2 ->im1
  '''
  img = np.copy(im1)
  # im1_cdf,bin_edges_im1 = getcdf(im1)
  # im2_cdf,bin_edges_im2 = getcdf(im2)
  inv_im2_cdf = inverse_transform_sampling(im2)
  # img = inv_im2_cdf(im1_cdf[im1])
  im_hist = histogramEqualization(im1)
  try:
    # img = inv_im2_cdf(im1_cdf[im1])
    img = inv_im2_cdf(im_hist)
  except Exception:
    print("Error Start")
    print(np.amin(im1))
    print(np.amax(im1))
    print(np.amin(im1_cdf))
    print(np.amax(im1_cdf))
    print(len(im1_cdf))
    # print(bin_edges_im2)  
    print(img.shape)
    print(im1.shape)
    print("yo",im1_cdf[im1])
    exit()
  # print("______")
  # print(im1.shape,im2.shape)
  # print(img.shape)
  # print("++++")
  return(img)


def steerablePyramids(img,levels,xlocal = []):
  if levels <= 0 or img.shape[0]<=1 or img.shape[1] <= 1:
    xlocal.append(img)
    return(xlocal)
  else:
    l = [cv2.filter2D(img,-1,ker) for ker in kernels]
    xlocal.append(l)
    steerablePyramids(cv2.pyrDown(img,(img.shape[1]//2,img.shape[0]//2)),levels-1,xlocal)
    return(xlocal)


def collapsePyramid(pyrr):
  currImg = pyrr[0]
  print("length",len(pyrr))
  for lev in range(len(pyrr) - 1):
    print("lev",lev)
    w = pyrr[lev+1][0].shape[1]
    h = pyrr[lev+1][0].shape[0]
    img = cv2.pyrUp(currImg,(w,h))
    sumIm = img
    for imm in pyrr[lev + 1]:
      sumIm = sumIm + imm
    currImg = sumIm
  return(currImg)

def matchTexture(noise,texture):
  noise = matchHistogram(noise,texture)
  analysis_pyr = steerablePyramids(texture,Levels,[])
  analysis_pyr.reverse()
  iters = 10
  for _ in range(iters):
    synthesis_pyramid = steerablePyramids(noise,Levels,[])
    synthesis_pyramid.reverse()
    for lev in range(1,Levels+1):
      for i in range(len(synthesis_pyramid[lev])):
        synthesis_pyramid[lev][i] = matchHistogram(synthesis_pyramid[lev][i],analysis_pyr[lev][i])
    synthesis_pyramid[0] = matchHistogram(synthesis_pyramid[0],analysis_pyr[0])
    # print("Second")
    # checkShapes(synthesis_pyramid,Levels)
    noise = collapsePyramid(synthesis_pyramid)
    noise = matchHistogram(noise,texture)
    print(noise)
  return(noise)

  


texture = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
noise = (np.random.rand(texture.shape[0],texture.shape[1])*255).astype(np.uint8)

# noise = (np.random.rand(9,9)*255).astype(np.uint8)
noiseOrg = np.copy(noise)

noise = noise/255.0
texture = texture/255.0


exptexture = matchTexture(noise,texture)


print(exptexture)
cv2.imshow("Original",texture)
cv2.imshow("NewTexture",exptexture)
# cv2.imshow("NewTexture",noise)
cv2.waitKey(0)
cv2.destroyAllWindows()



  

