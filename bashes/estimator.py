import numpy as np

class Estimator(object):
    """
    Top-level class for performing a Bayesian shear estimation.
    """
    def __init__(self,
        data,psfs,ivar,
        stampSize,featureMatrix = None):

        self.stampSize = stampSize
        self.nPixels = stampSize**2
        self.featureMatrix = featureMatrix
        if featureMatrix:
            # Check that the shape is compatible with the stamp size.
            assert featureMatrix.shape[1] == self.nPixels, (
                'Feature matrix has %d columns but stamps have %d pixels' % (
                    featureMatrix.shape[1],self.nPixels))
            self.nFeatures = featureMatrix.shape[0]
        else:
            self.nFeatures = self.nPixels

        # Save the data as feature vectors for each stamp.
        assert len(data.shape) in (2,3), 'Expected 2D or 3D input data'
        if len(data.shape) == 2:
            assert data.shape[0] % stampSize == 0 and data.shape[1] % stampSize == 0, (
                'Input data size not a multiple of the stamp size')
            ny,nx = data.shape[0]//stampSize,data.shape[1]//stampSize
            self.data = np.empty((nx*ny,self.nFeatures))
            for iy in range(ny):
                for ix in range(nx):
                    pixels = data[iy*stampSize:(iy+1)*stampSize,ix*stampSize:(ix+1)*stampSize].flat
                    if self.featureMatrix:
                        features = self.featureMatrix.dot(pixels)
                    else:
                        features = pixels
                    self.data[iy*nx+ix] = features
        else:
            # Handle a 3D array of data stamps...
            assert False,'3D data array not supported yet'

        # Save the PSF image for each stamp.
        if psfs.shape == (self.stampSize,self.stampSize):
            self.psfIsConstant = True
            self.psfs = psfs
        else:
            self.psfIsConstant = False
            # Handle a 3D or 2D array of PSF stamps...
            assert False,'Non-constant PSFs not supported yet'

        # Save the inverse variance vector for each stamp.
        try:
            self.ivar = float(ivar)
            self.ivarIsConstant = True
        except ValueError:
            assert False,'Non-constant ivar not supported yet'

    def usePrior(self,sourceStamp,fluxSigma,weight=1.):
        pass
