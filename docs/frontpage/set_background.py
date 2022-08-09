import sys
import matplotlib.pyplot as plt

from skimage.io import imread

#filename = sys.argv[1]
filename = "d_dz.png"
img = imread(filename)

plt.imshow(img)
plt.show()

pass