from HaarWaveletDenoise import *

# k is 0 - 1
def keepTopK(X,k):
  Y = np.copy(X)
  z = np.sort(np.abs(Y), axis=None)
  val = z[int((1-k)*(z.shape[0]-1))]
  Y[np.abs(Y) < val] = 0
  return(Y)


def infoLoss(X,k,level):
  size = min(X.shape[1],X.shape[0])
  if(size == 1 or level == 0):
    return(X)
  l = splitM(X)
  return(recombine(infoLoss(l[0],k,level - 1),keepTopK(l[1],k),keepTopK(l[2],k),keepTopK(l[3],k)))
  

def runlengthencoding(l):
  x = []
  rep = 1
  for i in range(len(l) - 1):
    if(l[i+1]==l[i]):
      rep = rep + 1
      if i == len(l) - 2:
        x.append([rep,l[i]])
    else:
      x.append([rep,l[i]])
      rep = 1
      if i == len(l) - 2:
        x.append([rep,l[i+1]])
  return(x)

def expand(l):
  x = []
  for y in l:
    x = x + y[0]*[y[1]]
  return(x)

def compress(img,levels,k):
  X = recTransform(img,levels)
  X = infoLoss(X,k,levels) #lossing information
  m,n = X.shape
  l = [m,n,levels]
  y = []
  for i in range(2*(n -1)+1):
    j = 0
    while(i >= m):
      i = i - 1
      j = j + 1
    while(i >= 0 and j<n):
      y.append(X[i][j])
      i = i - 1
      j = j + 1
  y = runlengthencoding(y)
  return(np.array(l+y))


def matify(l,m,n):
  X = np.zeros((m,n),dtype = np.float)
  idx = 0
  for i in range(2*(n -1)+1):
    j = 0
    while(i >= m):
      i = i - 1
      j = j + 1
    while(i >= 0 and j<n):
      X[i][j] = l[idx]
      i = i - 1
      j = j + 1
      idx += 1
  return(X)



def deCompress(X):
  M = X[0]
  N = X[1]
  levels = X[2]
  print(X.shape)
  l = expand(X[3:])
  Y = matify(l,M,N)
  Y = recInversetransform(Y,levels)
  return(Y)





img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
img = img/255.0
k = float(sys.argv[2])/100.0
levels = 4

comp = compress(img,levels,k)
compressed = deCompress(comp)

print("%d bytes" % (img.size * img.itemsize))
print("%d bytes" % (comp.size * comp.itemsize))

cv2.imshow('Original',img)
cv2.imshow('After Compression',compressed)
cv2.waitKey(0)
cv2.destroyAllWindows()

