import inspect
import numpy as np
import galsim
import bashes

import scipy.special # for component-wise erf

class Estimator(object):
    """
    Top-level class for performing a Bayesian shear estimation.
    """
    def __init__(self,
        data,psfs,ivar,
        stampSize,pixelScale,
        ntheta,nxy,xymax,nshear,gmax,
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

        # Precompute and save Dt.Cinv.D for each stamp
        D = self.data.reshape((self.ndata,self.nfeatures))
        self.DtCinvD = self.ivar*np.einsum('ce,ce->c',D,D)

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

    @staticmethod
    def addArgs(parser):
        """
        Add arguments to the provided command-line parser that support the fromArgs() method.
        """
        parser.add_argument('--ntheta', type = int, default = 16,
            help = 'Number of theta values to use in interpolation grid')
        parser.add_argument('--nxy', type = int, default = 7,
            help = 'Number of x,y values to use in interpolation grid')
        parser.add_argument('--xymax', type = float, default = 1.,
            help = 'Range of x,y to use in interpolation grid (pixels)')
        parser.add_argument('--nshear', type = int, default = 5,
            help = 'Number of reduced shear values for sampling the likelihood')
        parser.add_argument('--gmax', type = float, default = 0.06,
            help = 'Range of reduced shear for sampling the likelihood')

    @staticmethod
    def fromArgs(args):
        """
        Returns a dictionary of constructor parameter values based on the parsed args provided.
        """
        # Look up the named Estimator constructor parameters.
        pnames = (inspect.getargspec(Estimator.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        argsDict = vars(args)
        # Return a dictionary of constructor parameters provided in args.
        return { key:argsDict[key] for key in (set(pnames) & set(argsDict)) }

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
        # Calculate Mt.Cinv.M
        MtCinvM = self.ivar*np.einsum('abcde,abcde->abcd',M,M)
        # Calculate Dt.Cinv.M
        D = self.data.reshape((self.ndata,self.nfeatures))
        DtCinvM = self.ivar*np.einsum('ce,abcde->abcd',D,M)
        # Calculate chisq = (Dt-Mt).Cinv.(D-M) = Dt.Cinv.D - 2*Dt.Cinv.M + Mt.Cinv.M
        chiSq = self.DtCinvD - 2*DtCinvM + MtCinvM
        print 'chiSq shape is',chiSq.shape
        # Calculate gammaSq = 1 + r**2 MtCinvM
        r = fluxSigma/sourceModel.getFlux()
        rSq = r*r
        gammaSq = 1 + rSq*MtCinvM
        # Calculate phiSq = DtCinvD MtCinvM - DtCinvM**2
        phiSq = self.DtCinvD*MtCinvM - DtCinvM**2
        # Calculate the exponential arg psi
        psi = (chiSq + rSq*phiSq)/(2*gammaSq)
        # Calculate the normalization factor Gamma
        root2r = np.sqrt(2*rSq)
        gamma = np.sqrt(gammaSq)
        erfArg1 = (1 + rSq*DtCinvM)/(root2r*gamma)
        erfArg2 = 1./root2r
        Gamma = (1 + scipy.special.erf(erfArg1))/(1 + scipy.special.erf(erfArg2))/gamma
        # Calculate the negative log of the flux-integrated likelihood
        nll = psi - np.log(Gamma)
        print nll
