from HaarWaveletDenoise import *


img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
img = img/255.0

trans =  recTransform(img,8)

visualise(trans,8)
