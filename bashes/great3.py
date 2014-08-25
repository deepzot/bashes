import os
import os.path
import inspect
import numpy as np
from astropy.io import fits
import yaml
import galsim
import bashes

class Observation(object):
    """
    Represents a GREAT3 observation specified by a branch, index (0-199) and epoch.
    """
    def __init__(self,branch,index,epoch):
        """
        Initializes an observation using a branch path of the form 'control/ground/constant'
        that should be present under $GREAT3_ROOT (and also under $GREAT3_ROOT/truth
        if truth info is required), together with an image index (0-199) and an epoch number.
        Raises a RuntimeError if any problems are detected. After initialization, the following
        attributes are defined: nFields, nSubfieldsPerField, nEpochs, pixelScale, stampSize.
        """
        # Lookup the GREAT3 filesystem root.
        if 'GREAT3_ROOT' not in os.environ:
            raise RuntimeError('$GREAT3_ROOT is not set.')
        g3root = os.environ['GREAT3_ROOT']
        # Check for a valid branch path.
        pathNames = branch.split('/')
        if (len(pathNames) != 3 or
            pathNames[0] not in ('control','real_galaxy','variable_psf','multiepoch','full') or
            pathNames[1] not in ('ground','space') or
            pathNames[2] not in ('constant','variable')):
            raise RuntimeError('Invalid branch path: %r' % branch)
        # Lookup the path to this observation's branch.
        self.branchPath = os.path.join(g3root,branch)
        if not os.path.isdir(self.branchPath):
            raise RuntimeError('No such branch path: %r' % self.branchPath)
        # Do we have truth info available?
        self.truthPath = os.path.join(g3root,'truth',branch)
        if not os.path.isdir(self.truthPath):
            self.truthPath = None
        # Specify this branch's parameters.
        if pathNames[0] in ('variable_psf','full') or pathNames[2] == 'variable':
            self.nFields = 10
            self.nSubfieldsPerField = 20
        else:
            self.nFields = 200
            self.nSubfieldsPerField = 1
        if pathNames[0] in ('multiepoch','full'):
            self.nEpochs = 6
        else:
            self.nEpochs = 1
        if pathNames[1] == 'space':
            if self.nEpochs == 1:
                self.pixelScale = 0.05
                self.stampSize = 96
            else:
                self.pixelScale = 0.1
                self.stampSize = 48
        else:
            self.pixelScale = 0.2
            self.stampSize = 48
        # Check the index and epoch parameters.
        try:
            self.index = int(index)
            assert self.index >= 0 and self.index < 200
        except (ValueError,AssertionError):
            raise RuntimeError('Invalid branch index: %r' % index)
        try:
            self.epoch = int(epoch)
            assert self.epoch >= 0 and self.epoch < self.nEpochs
        except (ValueError,AssertionError):
            raise RuntimeError('Invalid branch epoch index: %r' % epoch)
        # Our galaxy and star images are loaded on demand.
        self.image = None
        self.stars = None
        # Our truth params and catalog are loaded on demand.
        self.truthParams = None
        self.truthCatalog = None

    @staticmethod
    def addArgs(parser):
        """
        Add arguments to the provided command-line parser that support the fromArgs() method.
        """
        parser.add_argument('--branch', type = str, default = 'control/ground/constant',
            help = 'Name of branch to use relative to $GREAT3_ROOT')
        parser.add_argument('--index', type = int, default = 0,
            help = 'Index of field to analyze (0-199)')
        parser.add_argument('--epoch', type = int, default = 0,
            help = 'Epoch number to analyze')

    @staticmethod
    def fromArgs(args):
        """
        Returns a dictionary of constructor parameter values based on the parsed args provided.
        """
        # Look up the named Estimator constructor parameters.
        pnames = (inspect.getargspec(Observation.__init__)).args[1:]
        # Get a dictionary of the arguments provided.
        argsDict = vars(args)
        # Return a dictionary of constructor parameters provided in args.
        return { key:argsDict[key] for key in (set(pnames) & set(argsDict)) }

    @classmethod
    def getGSParams(cls):
        if not hasattr(cls,'GSParams'):
            cls.GSParams = galsim.GSParams(maximum_fft_size=2**16)
        return cls.GSParams

    def getImage(self):
        """
        Returns the array of postage stamp image data for this observation and initializes
        our stampSize data member.
        """
        if self.image is None:
            dataStampsPath = os.path.join(self.branchPath,'image-%03d-%d.fits' % (
                self.index,self.epoch))
            hduList = fits.open(dataStampsPath)
            dataStamps = hduList[0].data
            hduList.close()
            # Check for the expected image dimensions.
            assert dataStamps.shape[0] == dataStamps.shape[1], 'Image data is not square'
            assert dataStamps.shape[0] == 100*self.stampSize, 'Image has unexpected dimensions'
            self.image = bashes.tiled.Tiled(dataStamps,self.stampSize)
            self.image.scale = self.pixelScale
        return self.image

    def getStars(self):
        """
        Returns the array of postage stamp starfield data for this observation.
        """
        if self.stars is None:
            psfStampsPath = os.path.join(self.branchPath,'starfield_image-%03d-%d.fits' % (
                self.index,self.epoch))
            hduList = fits.open(psfStampsPath)
            psfStamps = hduList[0].data
            hduList.close()
            self.stars = bashes.tiled.Tiled(psfStamps,self.stampSize)
            self.stars.scale = self.pixelScale
        return self.stars

    def getTruthParams(self):
        """
        Returns a dictionary of truth parameter values for this observation.
        """
        if self.truthParams is None:
            # No cached value available, so fetch it now.
            if not self.truthPath:
                raise RuntimeError('No truth available for observation')
            truthParamsPath = os.path.join(self.truthPath,'epoch_parameters-%03d-%d.yaml' % (
                self.index,self.epoch))
            with open(truthParamsPath,'r') as f:
                self.truthParams = yaml.load(f)
        return self.truthParams

    def getTruthCatalog(self):
        """
        Returns the truth catalog for this observation.
        """
        if self.truthCatalog is None:
            # No cached value available, so fetch it now.           
            if not self.truthPath:
                raise RuntimeError('No truth available for observation')
            truthCatalogPath = os.path.join(self.truthPath,'epoch_catalog-%03d-%d.fits' % (
                self.index,self.epoch))
            hduList = fits.open(truthCatalogPath)
            self.truthCatalog = hduList[1].data
            hduList.close()
        return self.truthCatalog

    def createSource(self,galaxyIndex,shifted = False,lensed = False):
        """
        Returns a GalSim model of the source for the specified galaxy index with
        optional centroid shifts and weak lensing distortion.
        """
        params = self.getTruthCatalog()[galaxyIndex]
        # Create the bulge component.
        bulge = galsim.Sersic(flux = params['bulge_flux'],
            half_light_radius = params['bulge_hlr'],
            n = params['bulge_n'], gsparams = Observation.getGSParams())
        bulge.applyShear(q = params['bulge_q'],
            beta = params['bulge_beta_radians']*galsim.radians)
        # Is there a disk component?
        if params['disk_flux'] > 0:
            disk = galsim.Exponential(flux = params['disk_flux'],
                half_light_radius = params['disk_hlr'],
                gsparams = Observation.getGSParams())
            disk.applyShear(q = params['disk_q'],
                beta = params['disk_beta_radians']*galsim.radians)
            source = galsim.Add(bulge,disk)
        else:
            source = bulge
        # Apply optional lensing.
        if lensed:
            source = source.lens(g1=params['g1'],g2=params['g2'],mu=params['mu'])
        # Apply optional centroid shift.
        if shifted:
            source = source.shift(
                dx=params['xshift']*self.pixelScale,
                dy=params['yshift']*self.pixelScale)
        return source

    def createPSF(self,galaxyIndex):
        """
        Returns a GalSim model of the PSF for the specified galaxy index.
        """
        catalog = self.getTruthCatalog()
        keys = catalog.columns.names
        params = catalog[galaxyIndex]
        # Create an empty list of models that will be convolved for the final PSF.
        models = [ ]
        # Add jitter contribution if provided.
        if 'opt_psf_jitter_sigma' in keys:
            jitterPSF = galsim.Gaussian(sigma=params['opt_psf_jitter_sigma']).shear(
                beta = params['opt_psf_jitter_beta']*galsim.degrees,
                e = params['opt_psf_jitter_e'])
            models.append(jitterPSF)
        # Add charge diffusion contribution if provided.
        if 'opt_psf_charge_sigma' in keys:
            chargePSF = galsim.Gaussian(sigma=params['opt_psf_charge_sigma']).shear(
                e1 = params['opt_psf_charge_e1'], e2 = 0.)
            models.append(chargePSF)
        # Create the optical component, which is always present.
        kmap = { 'opt_psf_lam_over_diam':'lam_over_diam', 'opt_psf_obscuration':'obscuration',
            'opt_psf_n_struts':'nstruts', 'opt_psf_strut_angle':'strut_angle',
            'opt_psf_pad_factor':'pad_factor', 'opt_psf_defocus':'defocus',
            'opt_psf_astig1':'astig1', 'opt_psf_astig2':'astig2', 'opt_psf_coma1':'coma1',
            'opt_psf_coma2':'coma2', 'opt_psf_trefoil1':'trefoil1', 'opt_psf_trefoil2':'trefoil2',
            'opt_psf_spher':'spher'}
        opticalPSFParams = { kmap[key]:params[key] for key in kmap }
        # Add units for the strut angle.
        opticalPSFParams['strut_angle'] *= galsim.degrees
        # Suppress warnings.
        opticalPSFParams['suppress_warning'] = True
        # Build the optical PSF from the params dictionary.
        models.append(galsim.OpticalPSF(**opticalPSFParams))
        # Add an atmospheric component if a FWHM value is provided.
        if 'atmos_psf_fwhm' in keys:
            atmosphericPSF = galsim.Kolmogorov(fwhm = params['atmos_psf_fwhm']).shear(
                beta = params['atmos_psf_beta']*galsim.degrees,
                e = params['atmos_psf_e'])
            models.append(atmosphericPSF)
        # Return the convolution of all PSF component models.
        return galsim.Convolve(models, gsparams = Observation.getGSParams())

    def createObject(self,galaxyIndex,shifted = True,lensed = True):
        """
        Returns a GalSim model of the object corresponding to the specified galaxy index,
        consisting of the source model with lensing distortion and centroid shift applied,
        and convolved with the appropriate PSF.
        """
        # Look up the component models.
        src = self.createSource(galaxyIndex,shifted,lensed)
        psf = self.createPSF(galaxyIndex)
        # Return their convolution.
        return galsim.Convolve(src,psf)

    def renderObject(self,galaxyIndex,shifted = True,lensed = True,addNoise = True):
        """
        Renders a postage stamp of the truth model for the specified galaxy index
        with optional noise (that will exactly match the noise used for GREAT3).
        """
        obj = self.createObject(galaxyIndex)
        stamp = bashes.utility.render(obj,self.pixelScale,size = self.stampSize)
        if addNoise:
            params = self.getTruthParams()
            seed = params['noise_seed']
            var = float(params['noise']['variance'])
            rng = galsim.BaseDeviate(seed = seed + galaxyIndex)
            noise = galsim.GaussianNoise(rng).withVariance(var)
            stamp.addNoise(noise)
        return stamp

def main():

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    Observation.addArgs(parser)
    parser.add_argument('--stamp', type = int, default = 0,
        help = 'index of stamp to use (0-99999)')
    parser.add_argument('--unlensed', action = 'store_true',
        help = 'do not include weak lensing effects')
    parser.add_argument('--test', action = 'store_true',
        help = 'test that stamp reconstructed from truth matches actual stamp')
    parser.add_argument('--ds9', action = 'store_true',
        help = 'display stamps in DS9')
    parser.add_argument('--save', type = str, default = None,
        help = 'save stamps to the specified FITS file')
    parser.add_argument('--truth', action = 'store_true',
        help = 'print catalog truth parameter values')
    args = parser.parse_args()

    # Initialize the requested observation.
    obs = Observation(**Observation.fromArgs(args))

    # Dump catalog truth info for this stamp if requested.
    if args.truth:
        import pprint
        catalog = obs.getTruthCatalog()
        truth = zip(catalog.columns.names,catalog[args.stamp])
        pprint.pprint(truth)

    # Render stamps if requested.
    if args.ds9 or args.save or args.test:
        lensed = not args.unlensed
        # Lookup the specified stamp's psf and source models.
        psfModel = obs.createPSF(args.stamp)
        srcModel = obs.createSource(args.stamp,shifted=True,lensed=lensed)

        # Render the PSF and source models separately.
        gsp = galsim.GSParams(maximum_fft_size = 2**16)
        psfStamp = bashes.utility.render(psfModel,obs.pixelScale,size=obs.stampSize)
        srcStamp = bashes.utility.render(srcModel,obs.pixelScale,size=obs.stampSize)

        # Render the combined object with and without noise.
        objStamp = obs.renderObject(args.stamp,shifted=True,lensed=lensed,addNoise=False)
        noiseStamp = obs.renderObject(args.stamp,shifted=True,lensed=lensed,addNoise=True)

        if args.test:
            dataStamp = obs.getImage().getStamp(args.stamp)

        if args.ds9:
            display = bashes.display.Display('cmap heat; scale sqrt')
            display.show(psfStamp)
            display.show(srcStamp)
            display.show(noiseStamp)
            if args.test:
                display.show(dataStamp,reuseLimits=True)
            display.show(objStamp,reuseLimits=True)

        if args.save:
            # Open this file using: ds9 -zoom to fit -color heat -multiframe <filename>
            stamps = [objStamp,psfStamp,srcStamp,noiseStamp]
            if args.test:
                stamps.append(dataStamp)
            galsim.fits.writeMulti(stamps, file_name = args.save)

        if args.test:
            delta = noiseStamp.array - dataStamp.array
            adiff = np.max(np.abs(delta))
            nonzero = dataStamp.array != 0
            rdiff = np.max(np.abs(delta[nonzero]/dataStamp.array[nonzero]))
            print 'Max difference between generated and saved stamps: %.3g (abs) %.3g (rel)' % (
                adiff,rdiff)
            noiseVar = obs.getTruthParams()['noise']['variance']
            print 'Std. deviation of differences / noise RMS = %.3g' % (
                np.std(delta)/np.sqrt(noiseVar))

if __name__ == "__main__":
    main()

