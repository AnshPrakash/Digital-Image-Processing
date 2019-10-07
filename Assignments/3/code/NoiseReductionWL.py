from HaarWaveletDenoise import *

def hardthersholding(X,t):
  Y = np.copy(X)
  Y[abs(Y) <= t] = 0
  return(Y)

def softthersholding(X,t):
  Y = np.copy(X)
  Y[abs(Y)<=t] = 0
  Y[Y > t] = Y[Y > t] - t
  Y[Y < - t] = Y[Y < -t] + t
  return(Y)




def recthersholding(X,level,hard,t):
  size = min(X.shape[1],X.shape[0])
  if(size == 1 or level == 0):
    return(X)
  l = splitM(X)
  if hard:
    return(recombine(recthersholding(l[0],level - 1,True,t),hardthersholding(l[1],t),hardthersholding(l[2],t),hardthersholding(l[3],t)))
  else:
    return(recombine(recthersholding(l[0],level - 1,False,t),softthersholding(l[1],t),softthersholding(l[2],t),softthersholding(l[3],t)))
  


def GaussNoise(m,n,mean,std):
  return(np.random.normal(mean,std,(m,n)))



img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
img = img/255.0

img = img + GaussNoise(img.shape[0],img.shape[1],0.0,0.1)
levels = 2
trans =  recTransform(img,levels)
visualise(trans,levels)
trans = recthersholding(trans,levels,False,0.3)
visualise(trans,levels)
proc =  recInversetransform(trans,levels)
cv2.imshow('Noised',img)
cv2.imshow('Noise reduced',proc)
cv2.waitKey(0)
cv2.destroyAllWindows()


