#!/usr/bin/env python

# Creates plots to illustrate the marginalization algorithm.

import os
import os.path
import argparse

import numpy as np
from astropy.io import fits
import yaml

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import galsim
import bashes

def main():

    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bashes.great3.Observation.addArgs(parser)
    parser.add_argument('--stamp', type = int, default = 0,
        help = 'Index of galaxy to analyze (0-9999')
    bashes.Estimator.addArgs(parser)
    args = parser.parse_args()

    # Check stamp arg.
    if args.stamp < 0 or args.stamp >= 10000:
        print 'stamp %d is outside of valid range 0-10000' % args.stamp
        return -1

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(**bashes.great3.Observation.fromArgs(args))

    # Load the one simulated postage stamp we will use to generate plots.
    dataStamps = obs.getImage()
    iy = args.stamp//dataStamps.ny
    ix = args.stamp%dataStamps.ny
    stamp = dataStamps.getStamp(ix,iy)

    # Create the PSF model for this stamp using truth info.
    psfModel = obs.createPSF(args.stamp)

    # Load the true noise variance used to simulate this epoch.
    params = obs.getTruthParams()
    noiseVarTruth = params['noise']['variance']

    # Build the estimator for this analysis (using only the first stamp, for now)
    estimator = bashes.Estimator(
        data=stamp,psfs=psfModel,ivar=1./noiseVarTruth,
        stampSize=obs.stampSize,pixelScale=obs.pixelScale,**bashes.Estimator.fromArgs(args))

    # Analyze stamps using the truth source prior for the first stamp.
    # Create an unlensed source prior for this stamp using truth info.
    prior = obs.createSource(args.stamp)
    estimator.usePrior(prior,fluxSigmaFraction = 0.1)

    # Initialize matplotlib.
    fig1 = plt.figure('fig1',figsize=(6,6))

if __name__ == '__main__':
    main()
