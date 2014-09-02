import numpy as np
import bashes

import scipy.ndimage
import scipy.interpolate

def fourierMatrix(n):
    """
    Returns the fourier transform matrix for a square 2d matrix of size n.
    """
    i,j = np.meshgrid(np.arange(n), np.arange(n))
    A = np.multiply.outer(i.flatten(), i.flatten())
    B = np.multiply.outer(j.flatten(), j.flatten())
    omega = np.exp(-2*np.pi*1J/n)
    return np.power(omega, A+B)

def circularize(t):
    """
    Returns the azimuthally averaged 2d image about the center. 
    """
    # average over theta
    sx, sy = t.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(X - sx/2 + 0.5, Y - sy/2 + 0.5)
    rbin = r.astype(np.int)
    tAvg = scipy.ndimage.mean(t, labels=rbin, index=np.arange(0, rbin.max()+1))
    # build 2d representation
    tAvgInterp = scipy.interpolate.InterpolatedUnivariateSpline(np.arange(len(tAvg)),tAvg)
    tAvg2d = tAvgInterp(r.flatten()).reshape(sx,sy)
    return tAvg2d

class FeatureCalculator(object):
    """
    Feature calculator abstract base class
    """
    def __init__(self, nfeatures):
        self.nfeatures = nfeatures
    def getFeatures(self, image, psf):
        pass

class FourierMoments(FeatureCalculator):
    """
    Calculates fourier domain moment features
    """
    def __init__(self, stampSize, pixelScale, sigma=None):
        self.stampSize = stampSize
        self.pixelScale = pixelScale
        self.sigma = sigma

        # k grid
        kx,ky = np.meshgrid(np.fft.fftfreq(stampSize),np.fft.fftfreq(stampSize))
        dk = kx[0,0]-kx[0,1]
        self.dksq = dk*dk

        # build part of weight function
        kxsq = kx*kx
        kysq = ky*ky
        self.ksq = kxsq + kysq
        sigmasqby2 = 0.5*sigma*sigma
        self.wg = np.exp(-self.ksq*sigmasqby2)

        # Build moment matrix
        moments = [np.ones((stampSize,stampSize)), 1J*kx, 1J*ky, self.ksq, kxsq - kysq, 2*kx*ky]
        self.M = np.array([m.flatten() for m in moments])

        nfeatures = self.M.shape[0]
        super(FourierMoments, self).__init__(nfeatures)

    def getFeatures(self, image, psf):
        """
        Returns the Fourier moment features of the provided image.
        """
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
        return self.dksq*self.M.dot(ftdeconvolved.flatten())

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

    def printFeature(label, feature):
        print '%10s %12.6g %12.6g' % (label, feature.real, feature.imag)

    printFeature('M_I', features[0])
    printFeature('M_x', features[1])
    printFeature('M_y', features[2])
    printFeature('M_r', features[3])
    printFeature('M_+', features[4])
    printFeature('M_x', features[5])

if __name__ == '__main__':
    main()
