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
  sum = 0
  for y in l:
    sum = sum + y[0]
  x = [0]*sum
  idx = 0
  for y in l:
    for _ in range(int(y[0])):
      x[idx] = y[1]
      idx = idx + 1
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





img3g = cv2.imread(sys.argv[1],cv2.IMREAD_UNCHANGED)
img3g = img3g/255.0
images = cv2.split(img3g)
compressed_img = []
for img in images:
  print(img.shape)
  k = float(sys.argv[2])/100.0
  levels = 3
  comp = compress(img,levels,k)
  compressed_img.append(comp)

recons = []
sizeOfCompres = 0
for comp in compressed_img:
  sizeOfCompres = sizeOfCompres + (comp.size * comp.itemsize)
  compressed = deCompress(comp)
  recons.append(compressed)


print("%d bytes" % (img3g.size * img3g.itemsize))
print("%d bytes" % (sizeOfCompres))



recons = cv2.merge(recons)

# img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
# print(img.shape)
# img = img/255.0
# k = float(sys.argv[2])/100.0
# levels = 3
# comp = compress(img,levels,k)
# compressed = deCompress(comp)

# print("%d bytes" % (img.size * img.itemsize))
# print("%d bytes" % (comp.size * comp.itemsize))
print("SNR: ",SNR(img3g,recons),"dB")
cv2.imshow('Original',img3g)
cv2.imshow('After Compression',recons)

# cv2.imwrite('Org.png',img3g*255)
# cv2.imwrite('Compressed.png',recons*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

