import os
import os.path
import numpy as np
from astropy.io import fits
import yaml
import galsim

class Observation(object):
    """
    Represents a GREAT3 observation specified by a branch, index and epoch.
    """
    def __init__(self,path,index,epoch):
        """
        Initializes a branch using a path of the form 'control/ground/constant'
        that should be present under $GREAT3_ROOT (and also under $GREAT3_ROOT/truth
        if truth info is required). Raises a RuntimeError if any problems are detected.
        """
        # Lookup the GREAT3 filesystem root.
        if 'GREAT3_ROOT' not in os.environ:
            raise RuntimeError('$GREAT3_ROOT is not set.')
        g3root = os.environ['GREAT3_ROOT']
        # Check for a valid branch path.
        pathNames = path.split('/')
        if (len(pathNames) != 3 or
            pathNames[0] not in ('control','real_galaxy','variable_psf','multiepoch','full') or
            pathNames[1] not in ('ground','space') or
            pathNames[2] not in ('constant','variable')):
            raise RuntimeError('Invalid branch path: %r' % path)
        # Lookup the path to this observation's branch.
        self.branchPath = os.path.join(g3root,path)
        if not os.path.isdir(self.branchPath):
            raise RuntimeError('No such branch path: %r' % self.branchPath)
        # Do we have truth info available?
        self.truthPath = os.path.join(g3root,'truth',path)
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
        # Our truth params and catalog are loaded on demand.
        self.truthParams = None
        self.truthCatalog = None

    def getImage(self):
        """
        Returns the array of postage stamp image data for this observation and initializes
        our stampSize data member.
        """
        dataStampsPath = os.path.join(self.branchPath,'image-%03d-%d.fits' % (
            self.index,self.epoch))
        hduList = fits.open(dataStampsPath)
        dataStamps = hduList[0].data
        hduList.close()
        # Check for the expected image dimensions.
        assert dataStamps.shape[0] == dataStamps.shape[1], 'Image data is not square'
        assert dataStamps.shape[0] == 100*self.stampSize, 'Image has unexpected dimensions'
        return dataStamps

    def getStars(self):
        """
        Returns the array of postage stamp starfield data for this observation.
        """
        psfStampsPath = os.path.join(self.branchPath,'starfield_image-%03d-%d.fits' % (
            self.index,self.epoch))
        hduList = fits.open(psfStampsPath)
        psfStamps = hduList[0].data
        hduList.close()
        return psfStamps

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

    def createSource(self,galaxyIndex):
        """
        Returns a GalSim model of the unlensed source for the specified galaxy index.
        """
        params = self.getTruthCatalog()[galaxyIndex]
        # Create the bulge component.
        bulge = galsim.Sersic(flux = params['bulge_flux'],
            half_light_radius = params['bulge_hlr'],
            n = params['bulge_n'])
        bulge.applyShear(q = params['bulge_q'],
            beta = params['bulge_beta_radians']*galsim.radians)
        # Is there a disk component?
        if params['disk_flux'] > 0:
            disk = galsim.Exponential(flux = params['disk_flux'],
                half_light_radius = params['disk_hlr'])
            disk.applyShear(q = params['disk_q'],
                beta = params['disk_beta_radians']*galsim.radians)
            source = galsim.Add(bulge,disk)
        else:
            source = bulge
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
        return galsim.Convolve(models)
