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
    def getFeatures(self, image, psf):
        """
        Returns a flat array of features that are linearly related to image pixels (given the psf).

        Args:
            image (np.ndarray): A 2D np array of image pixel values.
            psf (galsim.GSObject): The psf model for the corresponding image.
        """

    @abc.abstractmethod
    def getInverseCovariance(self, ivar):
        """
        Returns an inverse covariance matrix.
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
    def getFeatures(self, image, psf):
        """
        Returns a flat array of features that are linearly related to image pixels (given the psf).

        Args:
            image (np.ndarray): A 2D np array of image pixel values.
            psf (galsim.GSObject): The psf model for the corresponding image.
        """
        return image.flat

    def getInverseCovariance(self, ivar):
        """
        Returns an inverse covariance matrix.
        """
        if np.isscalar(ivar):
            return ivar*np.identity(self.nfeatures)
        elif ivar.shape == (self.stampSize, self.stampSize)
            return ivar.flat*np.identity(self.nfeatures)
        elif len(ivar.shape) == self.nfeatures
            return ivar*np.identity(self.nfeatures)
        else:
            assert False, 'Invalid ivar dimensions'

def fourierMatrix(n):
    """
    Returns the fourier transform matrix for a square 2d matrix of size n.

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

        # The fourier moments are complex numbers in general so multiply by two
        nfeatures = 2*self.M.shape[0]
        super(FourierMoments, self).__init__(nfeatures)

    def getFeatures(self, image, psf):
        """
        Returns a flat array of features that are linearly related to image pixels (given the psf).

        Args:
            image (np.ndarray): A 2D np array of image pixel values.
            psf (galsim.GSObject): The psf model for the corresponding image.
        """
        assert image.shape[0] == self.stampSize and image.shape[1] == self.stampSize, (
            'Input data has unexpected stamp size')
        # Fourier transform image
        ftimage = np.fft.fft2(image)

        # Render psf and fourier transform
        psf = bashes.utility.render(psf,scale=self.pixelScale,size=self.stampSize)
        ftpsf = np.fft.fft2(psf.array)

        # Build weight function
        if self.sigma is None:
            assert False,'FourierMoments: Non-constant sigma not supported yet'
            # estimate sigma from image
            #sigmasqby2 = 0.5*sigma*sigma
            #wg = np.exp(-self.ksq*sigmasqby2)
        else:
            wg = self.wg
        tSq = (np.conjugate(ftpsf)*ftpsf).real
        tSqCircle = np.fft.fftshift(circularize(np.fft.fftshift(tSq)))
        weight = wg*tSqCircle
        
        # Deconvolve image
        ftdeconvolved = ftimage / ftpsf * weight

        # Integral over ksq is essentially just a dot product between the moment matrix
        # and the deconvolved image
        complexFeatures = self.dksq*self.M.dot(ftdeconvolved.flatten())
        # split complex numbers and recombine into single list [real0, imag0, real1, imag1, ...]
        features = [j for i in zip(complexFeatures.real,complexFeatures.imag) for j in i]
        return features

    def getInverseCovariance(self, ivar):
        """
        Returns an inverse covariance matrix.
        """
        assert False, 'FourierMoments: Not Implemented'

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
    data = dataStamps.getStamp(0,0)

    # Load the constant PSF stamp to use for the analysis.    
    psf = obs.createPSF(0)

    featureCalc = FourierMoments(stampSize=stampSize,pixelScale=pixelScale,sigma=args.sigma)

    features = featureCalc.getFeatures(image=data.array,psf=psf)

    print '%10s %12s %12s' % ('feature', 'real', 'imag')        
    print '%10s %12.6g %12.6g' % ('M_I', features[0], features[1])
    print '%10s %12.6g %12.6g' % ('M_x', features[2], features[3])
    print '%10s %12.6g %12.6g' % ('M_y', features[4], features[5])
    print '%10s %12.6g %12.6g' % ('M_r', features[6], features[7])
    print '%10s %12.6g %12.6g' % ('M_+', features[8], features[9])
    print '%10s %12.6g %12.6g' % ('M_x', features[10], features[11])


if __name__ == '__main__':
    main()
