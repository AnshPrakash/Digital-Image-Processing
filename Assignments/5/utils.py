import matplotlib.pyplot as plt
import numpy as np

def show_images(images, level,cols = 1,titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.savefig("SteerablePyramid/level"+ str(level) +".png")
    plt.clf()

def displayPyramid(pyr,level):
  for lev in range(1,level+1):
    show_images(pyr[lev],lev)
  plt.imshow(pyr[0])
  plt.savefig("SteerablePyramid/level"+ str(0) +".png")

def gausianKernel(size,sigma,Nangles):
  angles = [ i*(3.14/Nangles) for i in range(Nangles+1)]
  cols = [np.array(list(range(size))) - (size-1)/2.0 for _ in range(size)]
  cols = np.array(cols)
  rows = np.copy(cols).T
  g = -np.exp(-(cols*cols + rows*rows)/2*sigma*sigma)/(2*3.14*(sigma**4))
  gx = rows*g
  gy = cols*g
  ga = [np.cos(alp)*gx + np.sin(alp)*gy for alp in angles]
  return(ga)


