#!/usr/bin/env python

import argparse
import math

import numpy as np

import galsim
import bashes

def main():

    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bashes.great3.Observation.addArgs(parser)
    bashes.Estimator.addArgs(parser)
    parser.add_argument('--ds9', action='store_true',
        help = 'Display data stamps using ds9')
    args = parser.parse_args()

    # initialize the optional display
    if args.ds9:
        display = bashes.Display('cmap heat; scale sqrt')
    else:
        display = None

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(**bashes.great3.Observation.fromArgs(args))

    # Load the postage stamps to analyze.
    dataStamps = obs.getImage()

    # Load the constant PSF stamp to use for the analysis.
    psfStamps = obs.getStars()

    # Load the true noise variance used to simulate this epoch.
    params = obs.getTruthParams()
    noiseVarTruth = params['noise']['variance']

    if display:
        # Display one data stamp.
        ix,iy = 25,75
        display.show(dataStamps.getStamp(ix,iy))
        # Display our reconstruction of the same data stamp.
        display.show(obs.renderObject(100*iy+ix,addNoise=True))
        return

    # Build the estimator for this analysis (using only the first stamp, for now)
    psfModel = obs.createPSF(0)
    estimator = bashes.Estimator(
        data=dataStamps.getStamp(0,0),psfs=psfModel,ivar=1./noiseVarTruth,
        stampSize=obs.stampSize,pixelScale=obs.pixelScale,**bashes.Estimator.fromArgs(args))

    # Analyze stamps using the truth source prior for the first stamp.
    prior = obs.createSource(0)
    priorFlux = prior.getFlux()
    estimator.usePrior(prior,fluxSigmaFraction = 0.1)

if __name__ == '__main__':
    main()
