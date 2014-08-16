import numpy as np
import galsim
import bashes

class Estimator(object):
    """
    Top-level class for performing a Bayesian shear estimation.
    """
    def __init__(self,
        data,psfs,ivar,
        stampSize,pixelScale,
        ntheta = 16, nxy = 7, xymax = 1.,
        nshear = 5, gmax = 0.06,
        featureMatrix = None):

        self.stampSize = stampSize
        self.pixelScale = pixelScale
        self.nPixels = stampSize**2
        self.featureMatrix = featureMatrix
        if featureMatrix:
            # Check that the shape is compatible with the stamp size.
            assert featureMatrix.shape[1] == self.nPixels, (
                'Feature matrix has %d columns but stamps have %d pixels' % (
                    featureMatrix.shape[1],self.nPixels))
            self.nfeatures = featureMatrix.shape[0]
        else:
            self.nfeatures = self.nPixels

        # Save the data as feature vectors for each stamp.
        assert len(data.shape) in (2,3), 'Expected 2D or 3D input data'
        if len(data.shape) == 2:
            assert data.shape[0] % stampSize == 0 and data.shape[1] % stampSize == 0, (
                'Input data size not a multiple of the stamp size')
            ny,nx = data.shape[0]//stampSize,data.shape[1]//stampSize
            self.ndata = nx*ny
            self.data = np.empty((self.ndata,self.nfeatures))
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
        if isinstance(psfs,galsim.GSObject):
            self.psfIsConstant = True
            self.psfs = psfs
        elif psfs.shape == (self.stampSize,self.stampSize):
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

        # Initialize our coarse theta grid in degrees. We do not include the endpoint
        # because of the assumed periodicity.
        self.ntheta = ntheta
        self.thetaGrid = np.linspace(0.,180.,ntheta,endpoint=False)

        # Initialize our x,y grid in pixels.
        self.nxy = nxy
        self.xyGrid = np.linspace(-xymax,+xymax,nxy)

        # Initialize our (g1,g2) grid.
        self.nshear = nshear
        self.shearGrid = np.linspace(-gmax,+gmax,nshear)

    def usePrior(self,sourceModel,fluxSigma,weight=1.):
        # Initialize float32 storage for the feature values we will calculate in parallel.
        assert self.ndata == 1,'ndata > 1 not supported yet'
        M = np.empty((self.ntheta,self.nshear**2,self.ndata,self.nxy**2,self.nfeatures),
            dtype=np.float32)
        print 'allocated %ld bytes for M' % M.nbytes
        # Loop over rotations.
        for ith,theta in enumerate(self.thetaGrid):
            # Loop over shears.
            for ig1,g1 in enumerate(self.shearGrid):
                for ig2,g2 in enumerate(self.shearGrid):
                    ig = ig1*self.nshear + ig2
                    print (ith,ig)
                    # Apply rotation and shear transforms.
                    transformed = sourceModel.rotate(theta*galsim.degrees).shear(g1=g1,g2=g2)
                    # Loop over PSF models (assuming we have a single PSF model for now)
                    idata = 0
                    convolved = galsim.Convolve(transformed,self.psfs)
                    # Loop over x,y shifts.
                    for ix,dx in enumerate(self.xyGrid):
                        for iy,dy in enumerate(self.xyGrid):
                            ixy = ix*self.nxy + iy
                            model = convolved.shift(dx=dx*self.pixelScale,dy=dy*self.pixelScale)
                            # Render the fully-specified model.
                            pixels = bashes.render(model,scale=self.pixelScale,size=self.stampSize)
                            if self.featureMatrix:
                                features = self.featureMatrix.dot(pixels.array.flat)
                            else:
                                features = pixels.array.flat
                            M[ith,ig,idata,ixy] = features
        # Calculate M.Cinv.M
        MCinvM = self.ivar*np.einsum('abcde,abcde->abcd',M,M)
        # Calculate D.Cinv.M
        D = self.data.reshape((self.ndata,self.nfeatures))
        DCinvM = self.ivar*np.einsum('ce,abcde->abcd',D,M)
        # Calculate chisq = ...
        print M.shape
        print D.shape
        print MCinvM.shape
        print DCinvM.shape
        # Calculate phi = ...
        # Calculate the flux-integrated likelihood
