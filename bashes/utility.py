import numpy as np
import galsim

def render(model,scale,size=None,stamp=None):
    """
    Returns a rendering of the specified GalSim model into a square postage stamp with the
    specified pixel scale (arcsecs). Either a stamp pixel size or an existing stamp (with its
    scale attribute set) must be provided. Note that the model will be convolved with the pixel
    response, so should not already be convolved with the pixel response.
    """
    if stamp is None:
        stamp = galsim.Image(size,size)
        stamp.scale = scale
    return model.drawImage(image = stamp)

def padPeriodic(x,n,offset=0.):
    """
    Extend the array x[i] with 0 <= i < N to cover -n <= i < N+n+1 assuming periodicity
    x[n+N] = x[n]+offset to duplicate existing entries. Returns a new array that contains
    a copy of the original array.  Note that an additional copy is needed on the high side
    to cover the interval x[N-1] - x[N].
    """
    nlo,nhi = n,n+1
    shape = list(x.shape)
    shape[0] += nlo+nhi
    padded = np.resize(x,tuple(shape))
    padded[:nlo] = x[-nlo:] - offset
    padded[nlo:-nhi] = x
    padded[-nhi:] = x[:nhi] + offset
    return padded

def getBinEdges(binCenters):
    """
    Returns a numpy array of bin edges corresponding to the input array of bin centers,
    which will have a length one greater than the input length. The outer edges will
    be the same as the outer bin centers. This function is normally used with the
    matplotlib pcolormesh function, e.g.

    ex,ey = getBinEdges(x),getBinEdges(y)
    plt.pcolormesh(ex,ey,z,cmap='rainbow')
    plt.contours(x,y,z,levels=(0,1,2),colors='w',linestyles=('-','--',':'))
    """
    edges = np.empty((len(binCenters)+1,),dtype=binCenters.dtype)
    edges[1:-1] = 0.5*(binCenters[1:] + binCenters[:-1])
    edges[0] = binCenters[0]
    edges[-1] = binCenters[-1]
    return edges
