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
    print repr(params['noise'])
    noiseVarTruth = params['noise']['variance']

    if display:
        ix,iy = 25,75
        # Display one data stamp.
        display.show(dataStamps.getStamp(ix,iy))
        # Display our reconstruction of the same data stamp.
        display.show(obs.renderObject(100*iy+ix,addNoise=True))
        return

    # Build the estimator for this analysis (using only the first stamp, for now)
    psfModel = obs.createPSF(0)
    estimator = bashes.Estimator(
        data=dataStamps.getStamp(0,0),psfs=psfModel,ivar=1./noiseVarTruth,
        stampSize=obs.stampSize,pixelScale=args.pixel_scale,**bashes.Estimator.fromArgs(args))

    # Analyze stamps using the truth source prior for the first stamp.
    prior = obs.createSource(0)
    priorFlux = prior.getFlux()
    estimator.usePrior(prior,fluxSigmaFraction = 0.1)

if __name__ == '__main__':
    main()
