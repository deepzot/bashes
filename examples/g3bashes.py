#!/usr/bin/env python

import argparse
import math

import numpy as np

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

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(args.branch,args.field,args.epoch)

    # Load the postage stamps to analyze.
    dataStamps = obs.getImage()

    # Load the constant PSF stamp to use for the analysis.
    psfStamps = obs.getStars()

    # Load the true noise variance used to simulate this epoch.
    params = obs.getTruthParams()
    noiseVarTruth = params['noise']['variance']

    # Load the per-galaxy true source properties used to simulate this epoch.
    truthCatalog = obs.getTruthCatalog()

    '''
    if display:
        # Display the PSF stamp.
        display.show(constPsf)
        psf = bashes.great3.createPSF(truthCatalog[0])
        psfStamp = bashes.render(psf,args.pixel_scale,obs.stampSize)
        display.show(psfStamp)
        # Display the first data stamp.
        firstStamp = dataStamps[:obs.stampSize,:obs.stampSize]
        display.show(firstStamp)
        # Display our reconstruction of the first data stamp.
        src = bashes.great3.createSource(truthCatalog[0])
        #srcStamp = bashes.render(src,args.pixel_scale,obs.stampSize)
        #display.show(srcStamp)
        transformed = src.shear(
            g1=truthCatalog[0]['g1'],g2=truthCatalog[0]['g2']
            ).shift(
            dx=truthCatalog[0]['xshift']*args.pixel_scale,
            dy=truthCatalog[0]['yshift']*args.pixel_scale)
        convolved = galsim.Convolve(transformed,psf)
        objStamp = bashes.render(convolved,args.pixel_scale,obs.stampSize)
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
        data=dataStamps[:obs.stampSize,:obs.stampSize],psfs=psfModel,ivar=1./noiseVarTruth,
        stampSize=obs.stampSize,pixelScale=args.pixel_scale,**bashes.Estimator.fromArgs(args))

    # Analyze stamps using the truth source prior for the first stamp.
    prior = bashes.great3.createSource(truthCatalog[0])
    priorFlux = prior.getFlux()
    estimator.usePrior(prior,fluxSigmaFraction = 0.1)

if __name__ == '__main__':
    main()
