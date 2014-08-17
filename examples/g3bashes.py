#!/usr/bin/env python

import os
import os.path
import argparse
import math

import numpy as np
from astropy.io import fits
import yaml

import galsim
import bashes

def main():

    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ds9', action = 'store_true',
        help = 'display results in DS9')
    parser.add_argument('--branch', type = str, default = 'control/ground/constant',
        help = 'Name of branch to use relative to $GREAT3_ROOT')
    parser.add_argument('--field', type = int, default = 0,
        help = 'Index of field to analyze (0-199)')
    parser.add_argument('--epoch', type = int, default = 0,
        help = 'Epoch number to analyze')
    parser.add_argument('--pixel-scale', type = float, default = 0.2,
        help = 'Pixel scale in arcsecs')
    bashes.Estimator.addArgs(parser)
    args = parser.parse_args()

    # initialize the optional display
    if args.ds9:
        display = bashes.Display('cmap heat; scale sqrt')
    else:
        display = None

    # Lookup the path to the GREAT3 branch we will use.
    if 'GREAT3_ROOT' not in os.environ:
        print '$GREAT3_ROOT is not set.'
        return -1
    branchPath = os.path.join(os.environ['GREAT3_ROOT'],args.branch)
    if not os.path.isdir(branchPath):
        print 'No such directory:',branchPath
        return -1

    # Load the postage stamps to analyze.
    if args.field < 0 or args.field >= 200:
        print 'Field index %d out of range 0-199.' % args.field
        return -1
    dataStampsPath = os.path.join(branchPath,'image-%03d-0.fits' % args.field)
    hduList = fits.open(dataStampsPath)
    dataStamps = hduList[0].data
    hduList.close()

    # Infer the stamp size from the stamp array shape.
    assert dataStamps.shape[0] == dataStamps.shape[1], 'Image data is not square'
    assert dataStamps.shape[0] % 100 == 0, 'Image data does not consist of 100x100 stamps'
    stampSize = dataStamps.shape[0]//100

    # Load the constant PSF stamp to use for the analysis.
    constPsfPath = os.path.join(branchPath,'starfield_image-%03d-0.fits' % args.field)
    hduList = fits.open(constPsfPath)
    constPsf = hduList[0].data[:stampSize,:stampSize]
    hduList.close()

    # Load the true noise variance used to simulate this epoch.
    truthParamsPath = os.path.join(os.environ['GREAT3_ROOT'],'truth',args.branch,
        'epoch_parameters-%03d-%d.yaml' % (args.field,args.epoch))
    with open(truthParamsPath,'r') as f:
        params = yaml.load(f)
        noiseVarTruth = params['noise']['variance']

    # Load the per-galaxy true source properties used to simulate this epoch.
    truthCatalogPath = os.path.join(os.environ['GREAT3_ROOT'],'truth',args.branch,
        'epoch_catalog-%03d-%d.fits' % (args.field,args.epoch))
    hduList = fits.open(truthCatalogPath)
    truthCatalog = hduList[1].data
    hduList.close()

    '''
    if display:
        # Display the PSF stamp.
        display.show(constPsf)
        psf = bashes.great3.createPSF(truthCatalog[0])
        psfStamp = bashes.render(psf,args.pixel_scale,stampSize)
        display.show(psfStamp)
        # Display the first data stamp.
        firstStamp = dataStamps[:stampSize,:stampSize]
        display.show(firstStamp)
        # Display our reconstruction of the first data stamp.
        src = bashes.great3.createSource(truthCatalog[0])
        #srcStamp = bashes.render(src,args.pixel_scale,stampSize)
        #display.show(srcStamp)
        transformed = src.shear(
            g1=truthCatalog[0]['g1'],g2=truthCatalog[0]['g2']
            ).shift(
            dx=truthCatalog[0]['xshift']*args.pixel_scale,
            dy=truthCatalog[0]['yshift']*args.pixel_scale)
        convolved = galsim.Convolve(transformed,psf)
        objStamp = bashes.render(convolved,args.pixel_scale,stampSize)
        display.show(objStamp)
        pulls = (firstStamp - objStamp.array)/math.sqrt(noiseVarTruth)
        display.show(pulls)
        import matplotlib.pyplot as plt
        plt.hist(pulls.flat,bins=40)
        plt.show()
        print 'RMS pull =',np.std(pulls.flat)
    '''

    # Build the estimator for this analysis (using only the first stamp, for now)
    psfModel = bashes.great3.createPSF(truthCatalog[0])
    estimator = bashes.Estimator(
        data=dataStamps[:stampSize,:stampSize],psfs=psfModel,ivar=1./noiseVarTruth,
        stampSize=stampSize,pixelScale=args.pixel_scale,**bashes.Estimator.fromArgs(args))

    # Analyze stamps using the truth source prior for the first stamp.
    prior = bashes.great3.createSource(truthCatalog[0])
    priorFlux = prior.getFlux()
    estimator.usePrior(prior,fluxSigmaFraction = 0.1)

if __name__ == '__main__':
    main()
