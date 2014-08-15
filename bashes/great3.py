import numpy as np

import galsim

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
