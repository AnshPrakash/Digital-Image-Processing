import numpy as np
import matplotlib.pyplot as plt
import sys
import pywt
import pywt.data
import cv2

# Load image
# original = pywt.data.camera()
original = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
original = original/255.0

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

cv2.imshow("1",LL)
cv2.imshow("2",LH)
cv2.imshow("3",HL)
cv2.imshow("4",HH)
cv2.waitKey(0)
cv2.destroyAllWindows()


