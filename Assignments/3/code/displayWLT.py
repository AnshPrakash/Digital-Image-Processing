from HaarWaveletDenoise import *


img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
img = img/255.0
trans =  recTransform(img,3)

visualise(trans,8)
# trans =  recInversetransform(trans,8)
# cv2.imshow('haar',trans)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

