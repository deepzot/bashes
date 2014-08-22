"""
Support for working with tiled images of postage stamps
"""

import numpy as np
import galsim

class Tiled(galsim.Image):
    """
    Represents a tiled image containing a grid of postage stamps.
    """
    def __init__(self,image,stampSize):
        """
        Initializes a tiled image for the specified observation from an existing
        Image or a numpy array.
        """
        if isinstance(image,galsim.Image):
            image = image.view()
        galsim.Image.__init__(self,image.astype(np.float32,copy=False),xmin=0,ymin=0)
        # Check that the image dimensions are compatible with the stamp size.
        h,w = self.array.shape
        if h % stampSize != 0 or w % stampSize != 0:
            raise RuntimeError('Tiled image size %r not compatible with stamp size %d' %
                ((w,h),stampSize))
        self.stampSize = stampSize
        self.nx = w//stampSize
        self.ny = h//stampSize

    def __iter__(self):
        """
        Returns an iterator over this tiled image's postage stamps.
        """
        for iy in range(self.ny):
            for ix in range(self.nx):
                yield self.getStamp(ix,iy)

    def checkCoords(self,ix,iy):
        """
        Raises a RuntimeError if (ix,iy) are out of bounds for this tiled image.
        """
        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            raise RuntimeError('Tiled image coordinates out of range %r' % (ix,iy))

    def getStampBounds(self,ix,iy):
        """
        Returns the bounding box for the stamp at tile coords (ix,iy).
        """
        # these explicit casts are necessary in case ix,iy are numpy.ints
        ix = int(ix)
        iy = int(iy)
        self.checkCoords(ix,iy)
        ssize = self.stampSize
        return galsim.BoundsI(ix*ssize, (ix+1)*ssize-1, iy*ssize, (iy+1)*ssize-1)

    def getStamp(self,ix,iy = None):
        """
        Returns the postage stamp at the tile coords (ix,iy). If iy is None, then ix
        is interpreted as iy*ny + ix.
        """
        if iy is None:
            iy = ix//self.ny
            ix = ix%self.ny
        bbox = self.getStampBounds(ix,iy)
        return self[bbox]

def create(stampSize,nx,ny,scale=None):
    """
    Creates a new Tiled image with the specified dimensions and pixels initialized to zero.
    """
    pixels = np.zeros((ny*stampSize,nx*stampSize),dtype=np.float32)
    img = Tiled(pixels,stampSize)
    if scale is not None:
        img.scale = scale
    return img
