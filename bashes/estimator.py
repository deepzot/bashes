import inspect
import numpy as np
import scipy.interpolate
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
        ntheta,nxy,xymax,
        xy_oversampling,theta_oversampling,
        ng,gmax,g1_center,g2_center,
        featureCalculator = None):

        self.stampSize = stampSize
        self.pixelScale = pixelScale
        self.nPixels = stampSize**2
        if featureCalculator is None:
            self.featureCalculator = bashes.features.PixelFeatures(stampSize)
        else:
            self.featureCalculator = featureCalculator

        # Save the PSF model for each stamp. For now we assume that psfs is an
        # iterable collection of galsim.GSObject models.
        try:
            assert len(psfs) == self.ndata, 'Expected same number of stamps and PSFs'
            for i,psf in enumerate(psfs):
                assert isinstance(psf,galsim.GSObject), 'PSF[%d] is not a GSObject' % i
            self.psfs = psfs
        except TypeError:
            raise RuntimeError('PSFs are not iterable')

        # Save the data as feature vectors for each stamp.
        if isinstance(data,galsim.Image):
            data = data.array
        assert len(data.shape) in (2,3), 'Expected 2D or 3D input data'
        if len(data.shape) == 2:
            assert data.shape[0] % stampSize == 0 and data.shape[1] % stampSize == 0, (
                'Input data size not a multiple of the stamp size')
            ny,nx = data.shape[0]//stampSize,data.shape[1]//stampSize
            self.ndata = nx*ny
            self.data = np.empty((self.ndata,self.featureCalculator.nfeatures))
            for iy in range(ny):
                for ix in range(nx):
                    pixels = data[iy*stampSize:(iy+1)*stampSize,ix*stampSize:(ix+1)*stampSize]
                    features = self.featureCalculator.getFeatures(pixels,psfs[iy*nx+ix])
                    self.data[iy*nx+ix] = features
        else:
            # Handle a 3D array of data stamps...
            assert data.shape[1] == stampSize and data.shape[2] == stampSize, (
                'Input data has unexpected stamp size')
            self.ndata = data.shape[0]
            self.data = np.empty((self.ndata,self.featureCalculator.nfeatures))
            for i in range(self.ndata):
                pixels = data[i]
                features = self.featureCalculator.getFeatures(pixels,psfs[i])
                self.data[i] = features

        # Save the inverse variance vector for each stamp.
        try:
            self.ivar = float(ivar)
            self.ivarIsConstant = True
        except ValueError:
            assert False,'Non-constant ivar not supported yet'

        # Precompute and save Dt.Cinv.D for each stamp
        D = self.data.reshape((self.ndata,self.featureCalculator.nfeatures))
        self.DtCinvD = self.ivar*np.einsum('ce,ce->c',D,D)
        # Reshape for broadcasting over MtCinvM.
        self.DtCinvD = self.DtCinvD.reshape((1,1,self.ndata,1))

        # Initialize our coarse theta grid in degrees. We do not include the endpoint
        # because of the assumed periodicity.
        self.ntheta = ntheta
        self.thetaGrid = np.linspace(0.,180.,ntheta,endpoint=False)

        # Initialize a corresponding grid suitable for periodic interpolation.
        self.thetaOrder = 3
        self.thetaPeriodic = bashes.utility.padPeriodic(self.thetaGrid,self.thetaOrder,180.)

        # Initialize our oversampled theta grid in degrees.
        self.thetaFine = np.linspace(0.,180.,ntheta*theta_oversampling,endpoint=False)

        # Initialize our x,y grid in pixel units.
        self.nxy = nxy
        self.xyGrid = np.linspace(-xymax,+xymax,nxy)

        # Initialize our oversampled x,y grid in pixel units.
        nxyFine = (nxy-1)*xy_oversampling
        dxyFine = 2*xymax/nxyFine
        self.xyFine = np.linspace(-xymax + 0.5*dxyFine,+xymax - 0.5*dxyFine,nxyFine)

        # Initialize our (g1,g2) grid.
        dg = np.linspace(-gmax,+gmax,ng)
        g1,g2 = np.meshgrid(g1_center+dg,g2_center+dg)
        self.g1vec = g1.flatten()
        self.g2vec = g2.flatten()
        self.nshear = len(self.g1vec)

        # Initialize float32 storage for the feature values we will calculate in parallel.
        self.M = np.empty((self.ntheta,self.nshear,self.ndata,self.nxy**2,self.featureCalculator.nfeatures),
            dtype=np.float32)
        # Initialize storage for marginalized NLL arrays.
        self.nllTheta = np.empty((self.ntheta,self.nshear,self.ndata))
        self.nll = np.empty((self.nshear,self.ndata))
        print 'allocated %ld bytes for M' % self.M.nbytes

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
        parser.add_argument('--xy-oversampling', type = int, default = 6,
            help = 'Amount of interpolated oversampling to use in x,y (1 = none)')
        parser.add_argument('--theta-oversampling', type = int, default = 32,
            help = 'Amount of interpolated oversampling to use in theta (1 = none)')
        parser.add_argument('--ng', type = int, default = 5,
            help = 'Number of g1,g2 values for sampling the likelihood')
        parser.add_argument('--gmax', type = float, default = 0.06,
            help = 'Range of reduced shear for sampling the likelihood')
        parser.add_argument('--g1-center', type = float, default = 0.,
            help = 'g1 value at center of shear grid')
        parser.add_argument('--g2-center', type = float, default = 0.,
            help = 'g2 value at center of shear grid')

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

    def usePrior(self,sourceModel,fluxSigmaFraction,weight=1.,traceMsg=None):
        # Loop over rotations.
        for ith,theta in enumerate(self.thetaGrid):
            # Loop over shears.
            for ig,(g1,g2) in enumerate(zip(self.g1vec,self.g2vec)):
                if traceMsg:
                    print traceMsg % (ith,ig)
                # Apply rotation and shear transforms.
                transformed = sourceModel.rotate(theta*galsim.degrees).shear(g1=g1,g2=g2)
                # Loop over PSF models for each data stamp.
                for idata,psf in enumerate(self.psfs):
                    convolved = galsim.Convolve(transformed,self.psfs[idata])
                    # Loop over x,y shifts.
                    for iy,dy in enumerate(self.xyGrid):
                        for ix,dx in enumerate(self.xyGrid):
                            ixy = iy*self.nxy + ix
                            model = convolved.shift(dx=dx*self.pixelScale,dy=dy*self.pixelScale)
                            # Render the fully-specified model.
                            pixels = bashes.utility.render(model,scale=self.pixelScale,size=self.stampSize)
                            features = self.featureCalculator.getFeatures(pixels.array)
                            self.M[ith,ig,idata,ixy] = features
        # Calculate Mt.Cinv.M
        MtCinvM = self.ivar*np.einsum('abcde,abcde->abcd',self.M,self.M)
        # Calculate Dt.Cinv.M
        D = self.data.reshape((self.ndata,self.featureCalculator.nfeatures))
        DtCinvM = self.ivar*np.einsum('ce,abcde->abcd',D,self.M)
        # Calculate chisq = (Dt-Mt).Cinv.(D-M) = Dt.Cinv.D - 2*Dt.Cinv.M + Mt.Cinv.M
        chiSq = self.DtCinvD - 2*DtCinvM + MtCinvM
        # Calculate gammaSq = 1 + r**2 MtCinvM
        rSq = fluxSigmaFraction**2
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
        self.nllXYTheta = psi - np.log(Gamma)
        # Loop over shears and data stamps to perform marginalization integrals.
        for ig in range(self.nshear):
            for idata in range(self.ndata):
                # Marginalize nll(x,y,theta) over (x,y) for each theta.
                for ith in range(self.ntheta):
                    # Tabulate nll(x,y) on the oversampled grid.
                    nllFine = self.getNllXYFine(ith,ig,idata)
                    # Estimate -log of the integral over x,y.
                    self.nllTheta[ith,ig,idata] = Estimator.marginalize(nllFine)
                # Tabulate nll(theta) using a periodic spline interpolation.
                nllFine = self.getNllFine(ig,idata)
                # Estimate -log of the integral over theta.
                self.nll[ig,idata] = Estimator.marginalize(nllFine)

    @staticmethod
    def marginalize(nll):
        """
        Returns an estimate of the -log of the integral of exp(-nll) using the provided
        values of nll tabulated on a uniform grid over the integration domain.
        """
        nllMin = np.min(nll)
        expSum = np.sum(np.exp(nllMin - nll))
        return nllMin - np.log(expSum/np.size(nll))

    def getNllXYFine(self,ith,ig,idata):
        """
        Returns an (x,y) array of NLL values marginalized over flux for the specified
        theta, shear, and data stamp index. Values are interpolated onto the fine
        oversampled grid self.xyFine used for (x,y) marginalization.
        """
        # Calculate a quadratic 2D spline of the tabulated nll(x,y,theta) at this theta.
        nllXY = self.nllXYTheta[ith,ig,idata].reshape((self.nxy,self.nxy))
        spline = scipy.interpolate.RectBivariateSpline(
            self.xyGrid,self.xyGrid,nllXY,kx=2,ky=2,s=0.)
        # Evaluate and return the spline on our oversampled fine grid.
        return spline(self.xyFine,self.xyFine)

    def getNllFine(self,ig,idata):
        """
        Returns a (theta) array of NLL values marginalized over flux,x,y for the
        specified shear and data stamp index. Values are interpolated onto the fine
        oversampled grid self.thetaFine used for (theta) marginalization.
        """
        nllPeriodic = bashes.utility.padPeriodic(self.nllTheta[:,ig,idata],self.thetaOrder)
        spline = scipy.interpolate.interp1d(self.thetaPeriodic,nllPeriodic,
            kind=self.thetaOrder,copy=False)
        return spline(self.thetaFine)
