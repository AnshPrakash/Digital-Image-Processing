import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
  
'''
  All Images are of power of 2
'''

def splitM(X):
  upper_half = np.hsplit(np.vsplit(X, 2)[0], 2)
  lower_half = np.hsplit(np.vsplit(X, 2)[1], 2)
  upper_left = upper_half[0]
  upper_right = upper_half[1]
  lower_left = lower_half[0]
  lower_right = lower_half[1]
  return([upper_left,upper_right,lower_left,lower_right])

def recombine(c11,c12,c21,c22):
  C = np.vstack([np.hstack([c11, c12]), np.hstack([c21, c22])])
  return(C)

def horizontal(X):
  hozHar = np.copy(X)
  size = int(X.shape[1]/2)
  for i in range(size):
    hozHar[:,i] = (X[:,2*i] + X[:,2*i + 1])/(2**0.5)
    hozHar[:,size + i] = (X[:,2*i] - X[:,2*i + 1])/(2**0.5)
  return(hozHar)

def vertical(X):
  vertHar = np.copy(X)
  size = int(X.shape[0]/2)
  for i in range(size):
    vertHar[i] = (X[2*i] + X[2*i + 1])/(2**0.5)
    vertHar[size + i] = (X[2*i] - X[2*i + 1])/(2**0.5)
  return(vertHar)

def HaarTransform(X):
  '''
    returns a  list of wavelet transform components
  '''
  VTransf = vertical(X)
  haartransf = horizontal(VTransf)
  return(splitM(haartransf))


def ihoztransform(X):
  iX = np.copy(X)
  size = iX.shape[1]
  # even
  for i in range(0,size,2):
    iX[:,i] = (X[:,int(i/2)] + X[:,int(size/2) + int(i/2)])/(2**0.5)
  # odd
  for i in range(1,size,2):
    iX[:,i] = (X[:,int(i/2)] - X[:,int(size/2) + int(i/2)])/(2**0.5)
  return(iX)

def ivertransform(X):
  iX = np.copy(X)
  size = iX.shape[0]
  # even
  for i in range(0,size,2):
    iX[i] = (X[int(i/2)] + X[int(size/2) + int(i/2)])/(2**0.5)
  # odd
  for i in range(1,size,2):
    iX[i] = (X[int(i/2)] - X[int(size/2) + int(i/2)])/(2**0.5)
  return(iX)


def inverseHaartransform(X):
  '''
    gets the wavelet transform in image form and returns a numpy array
  '''
  iX = ihoztransform(X)
  iX = ivertransform(iX)
  return(iX)


def recTransform(X,level):
  size = min(X.shape[1],X.shape[0])
  if(size == 1 or level == 0):
    return(X)
  l = HaarTransform(X)
  return(recombine(recTransform(l[0],level-1) ,l[1],l[2],l[3]))


def recInversetransform(X,level):
  size = min(X.shape[1],X.shape[0])
  if(size == 1 or level == 1):
    return(inverseHaartransform(X))
  l = splitM(X)
  return(inverseHaartransform(recombine(recInversetransform(l[0],level-1) ,l[1],l[2],l[3])))



def scaleMat(X):
  maxE = np.amax(X)
  minE = np.amin(X)
  if (maxE - minE == 0):
    return(X/255.0) #just a choice
  return((X - minE)/(maxE - minE))


def displayable(X,level):
  size = min(X.shape[1],X.shape[0])
  if(size == 1 or level == 0):
    return(scaleMat(X))
  l = splitM(X)
  for i in range(1,4):
    l[i] = scaleMat(l[i])
  return(recombine(displayable(l[0],level-1),l[1],l[2],l[3]))



def visualise(X,level):
  Y = displayable(X,level)
  cv2.imshow('visualise Haar',Y)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
# img = img/255.0



# Wavelet transform of image, and plot approximation and details

# coeffs2 = HaarTransform(img)
# LL,LH, HL, HH = coeffs2
# fig = plt.figure(figsize=(12, 3))
# for i, a in enumerate([LL,LH,HL,HH]):
#     ax = fig.add_subplot(1, 4, i + 1)
#     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     ax.set_title(titles[i], fontsize=10)
#     ax.set_xticks([])
#     ax.set_yticks([])

# fig.tight_layout()
# plt.show()
# rc =  recombine(LL,LH, HL, HH)
# rc =  inverseHaartransform(recombine(LL,LH, HL, HH))
# print(img.shape)
# print(rc.shape)
# cv2.imshow('image',img)
# cv2.imshow('haar',rc)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# trans =  recTransform(img,8)
# trans =  recInversetransform(trans,8)
# cv2.imshow('haar',trans)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

