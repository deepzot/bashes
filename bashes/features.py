import abc 
import numpy as np
import bashes

import scipy.ndimage
import scipy.interpolate

class AbsFeatureCalculator(object):
    """
    Feature calculator abstract base class.

    Attributes:
        nfeatures (int): The number of features to calculate.

    Args:
        nfeatures (int): The number of features to calculate.
    """
    def __init__(self, nfeatures):
        self.nfeatures = nfeatures

    @abc.abstractmethod
    def getFeatureTransform(self, psf):
        """
        Returns a matrix that transforms pixel values to features.

        Args:
            psf (galsim.GSObject): The psf model for the corresponding image.
        """

class PixelFeatures(AbsFeatureCalculator):
    """
    Simple pixel feature calculator.

    Attributes:
        nfeatures (int): The total number of pixels.

    Args:
        stampSize (int): The postage stamp size in pixels.
    """
    def __init__(self, stampSize):
        self.stampSize = stampSize
        nfeatures = stampSize*stampSize
        super(PixelFeatures, self).__init__(nfeatures)
    def getFeatureTransform(self, psf):
        """
        Returns a matrix that transforms pixel values to features.

        Args:
            psf (galsim.GSObject): The psf model for the corresponding image.
        """
        npixels = self.stampSize*self.stampSize
        assert npixels == self.nfeatures
        return np.identity(npixels)

def fourierMatrix(n):
    """
    Returns the fourier transform matrix for a square 2d matrix of size n.
    The following are equivalent:

        fourierMatrix(self.stampSize).dot(image.flatten())
        np.fft.fft2(image).flatten()

    Args:
        n (int): Square image size.
    """
    i,j = np.meshgrid(np.arange(n), np.arange(n))
    A = np.multiply.outer(i.flatten(), i.flatten())
    B = np.multiply.outer(j.flatten(), j.flatten())
    omega = np.exp(-2*np.pi*1J/n)
    return np.power(omega, A+B)

def circularize(image):
    """
    Returns the azimuthally averaged 2d image about the center. 

    Args:
        image (np.ndarray): Square 2D image.
    """
    # average over theta
    sx, sy = image.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(X - sx/2 + 0.5, Y - sy/2 + 0.5)
    rbin = r.astype(np.int)
    avg = scipy.ndimage.mean(image, labels=rbin, index=np.arange(0, rbin.max()+1))
    # build 2d representation
    avgInterp = scipy.interpolate.InterpolatedUnivariateSpline(np.arange(len(avg)),avg)
    avg2d = avgInterp(r.flatten()).reshape(sx,sy)
    return avg2d

class FourierMoments(AbsFeatureCalculator):
    """
    Calculates fourier domain moment features.

    Args:
        stampSize (int): The postage stamp size in pixels.
        pixelScale (int): The postage stamp pixel scale.
        sigma (float,None): The Gaussian weight function sigma. A value of None 
            indicates that sigma should be calculated on-the-fly (not supported yet).
    """
    def __init__(self, stampSize, pixelScale, sigma):
        self.stampSize = stampSize
        self.pixelScale = pixelScale
        self.sigma = sigma

        # k grid
        kx,ky = np.meshgrid(np.fft.fftfreq(stampSize),np.fft.fftfreq(stampSize))
        dk = kx[0,0]-kx[0,1]
        self.dksq = dk*dk

        # build Gaussian part of weight function
        kxsq = kx*kx
        kysq = ky*ky
        self.ksq = kxsq + kysq
        if sigma is None:
            assert False,'FourierMoments: Non-constant sigma not supported yet'
        else:
            sigmasqby2 = 0.5*sigma*sigma
            self.wg = np.exp(-self.ksq*sigmasqby2)

        # Build moment matrix
        moments = [np.ones((stampSize,stampSize)), 1J*kx, 1J*ky, self.ksq, kxsq - kysq, 2*kx*ky]
        self.M = np.array([m.flatten() for m in moments])

        # The fourier moments are complex numbers so multiply by two
        nfeatures = 2*self.M.shape[0]
        super(FourierMoments, self).__init__(nfeatures)

    def getFeatureTransform(self, psf):
        """
        Returns a matrix that transforms pixel values to features.

        Args:
            psf (galsim.GSObject): The psf model for the corresponding image.
        """
        # Fourier transform matrix
        DFT = fourierMatrix(self.stampSize)

        # Render psf and fourier transform
        psf = bashes.utility.render(psf,scale=self.pixelScale,size=self.stampSize)
        psfDFT = DFT.dot(psf.array.flatten())

        # Build weight function
        if self.sigma is None:
            assert False,'FourierMoments: Non-constant sigma not supported yet'
            # estimate sigma from image
            #sigmasqby2 = 0.5*sigma*sigma
            #wg = np.exp(-self.ksq*sigmasqby2)
        else:
            wg = self.wg
        tSq = (np.conjugate(psfDFT)*psfDFT).reshape(self.stampSize,self.stampSize).real
        tSqCircle = np.fft.fftshift(circularize(np.fft.fftshift(tSq)))
        weight = wg*tSqCircle

        I = np.identity(self.stampSize*self.stampSize)
        integrand = I*weight.flatten()/psfDFT

        # Integral over ksq is essentially just a dot product between the moment matrix
        # and the deconvolved image
        A = (self.dksq*self.M).dot(integrand).dot(DFT)

        # split real and imag parts into separate features
        assert 2*A.shape[0] == self.nfeatures, 'Invalid number of features'
        Afinal = np.empty((self.nfeatures,A.shape[1]))
        Afinal[::2,:] = A.real
        Afinal[1::2,:] = A.imag

        return Afinal

def main():
    import argparse
    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sigma', type = float, default = 1.,
        help = 'Weight function size')
    bashes.great3.Observation.addArgs(parser)
    args = parser.parse_args()

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(**bashes.great3.Observation.fromArgs(args))
    stampSize = obs.stampSize
    pixelScale = obs.pixelScale

    # Load the postage stamps to analyze.
    dataStamps = obs.getImage()
    stamp = dataStamps.getStamp(0,0)

    # Load the constant PSF stamp to use for the analysis.    
    psf = obs.createPSF(0)

    featureCalc = FourierMoments(stampSize=stampSize,pixelScale=pixelScale,sigma=args.sigma)

    A = featureCalc.getFeatureTransform(psf=psf)

    features = A.dot(stamp.array.flatten())

    print '%10s %12s %12s' % ('feature', 'real', 'imag')        
    print '%10s %12.6g %12.6g' % ('M_I', features[0], features[1])
    print '%10s %12.6g %12.6g' % ('M_x', features[2], features[3])
    print '%10s %12.6g %12.6g' % ('M_y', features[4], features[5])
    print '%10s %12.6g %12.6g' % ('M_r', features[6], features[7])
    print '%10s %12.6g %12.6g' % ('M_+', features[8], features[9])
    print '%10s %12.6g %12.6g' % ('M_x', features[10], features[11])


if __name__ == '__main__':
    main()
