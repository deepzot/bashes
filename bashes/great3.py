import os
import os.path
import numpy as np
from astropy.io import fits
import yaml
import galsim

class Observation(object):
    """
    Represents a GREAT3 observation specified by a branch, field and epoch.
    """
    def __init__(self,path,field,epoch):
        """
        Initializes a branch using a path of the form 'control/ground/constant'
        that should be present under $GREAT3_ROOT (and also under $GREAT3_ROOT/truth
        if truth info is required). Raises a RuntimeError if any problems are detected.
        """
        if 'GREAT3_ROOT' not in os.environ:
            raise RuntimeError('$GREAT3_ROOT is not set.')
        g3root = os.environ['GREAT3_ROOT']
        self.branchPath = os.path.join(g3root,path)
        if not os.path.isdir(self.branchPath):
            raise RuntimeError('No such branch path: %r' % self.branchPath)
        # Do we have truth info available?
        self.truthPath = os.path.join(g3root,'truth',path)
        if not os.path.isdir(self.truthPath):
            self.truthPath = None
        # Check the field and epoch parameters.
        try:
            self.field = int(field)
            assert self.field >= 0 and self.field < 200
        except (ValueError,AssertionError):
            raise RuntimeError('Invalid branch field index: %r' % field)
        try:
            self.epoch = int(epoch)
            assert self.epoch >= 0
        except (ValueError,AssertionError):
            raise RuntimeError('Invalid branch epoch index: %r' % epoch)

    def getImage(self):
        """
        Returns the array of postage stamp image data for this observation and initializes
        our stampSize data member.
        """
        dataStampsPath = os.path.join(self.branchPath,'image-%03d-%d.fits' % (
            self.field,self.epoch))
        hduList = fits.open(dataStampsPath)
        dataStamps = hduList[0].data
        hduList.close()
        # Infer the stamp size from the stamp array shape.
        assert dataStamps.shape[0] == dataStamps.shape[1], 'Image data is not square'
        assert dataStamps.shape[0] % 100 == 0, 'Image data does not consist of 100x100 stamps'
        self.stampSize = dataStamps.shape[0]//100
        return dataStamps

    def getStars(self):
        """
        Returns the array of postage stamp starfield data for this observation.
        """
        psfStampsPath = os.path.join(self.branchPath,'starfield_image-%03d-%d.fits' % (
            self.field,self.epoch))
        hduList = fits.open(psfStampsPath)
        psfStamps = hduList[0].data
        hduList.close()
        return psfStamps

    def getTruthParams(self):
        """
        Returns a dictionary of truth parameter values for this observation.
        """
        if not self.truthPath:
            raise RuntimeError('No truth available for observation')
        truthParamsPath = os.path.join(self.truthPath,'epoch_parameters-%03d-%d.yaml' % (
            self.field,self.epoch))
        with open(truthParamsPath,'r') as f:
            params = yaml.load(f)
        return params

    def getTruthCatalog(self):
        """
        Returns the truth catalog for this observation.
        """       
        if not self.truthPath:
            raise RuntimeError('No truth available for observation')
        truthCatalogPath = os.path.join(self.truthPath,'epoch_catalog-%03d-%d.fits' % (
            self.field,self.epoch))
        hduList = fits.open(truthCatalogPath)
        truthCatalog = hduList[1].data
        hduList.close()
        return truthCatalog

def createSource(params):
    """
    Returns a GalSim model of the unlensed source specified by params.
    """
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

def createPSF(params):
    """
    Returns a GalSim model of the optical + atmospheric PSF specified by params.
    """
    # Create the optical component.
    opticalPSF = galsim.OpticalPSF(lam_over_diam = params['opt_psf_lam_over_diam'],
        obscuration = params['opt_psf_obscuration'],
        nstruts = params['opt_psf_n_struts'],
        strut_angle = params['opt_psf_strut_angle'],
        pad_factor = params['opt_psf_pad_factor'],
        defocus = params['opt_psf_defocus'],
        astig1 = params['opt_psf_astig1'],
        astig2 = params['opt_psf_astig2'],
        coma1 = params['opt_psf_coma1'],
        coma2 = params['opt_psf_coma2'],
        trefoil1 = params['opt_psf_trefoil1'],
        trefoil2 = params['opt_psf_trefoil2'],
        spher = params['opt_psf_spher'])
    atmosphericPSF = galsim.Kolmogorov(fwhm = params['atmos_psf_fwhm'])
    atmosphericPSF.applyShear(e = params['atmos_psf_e'],
        beta = params['atmos_psf_beta']*galsim.degrees)
    PSF = galsim.Convolve(opticalPSF,atmosphericPSF)
    return PSF
