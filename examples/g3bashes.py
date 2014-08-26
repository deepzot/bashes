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
    parser.add_argument('--save', type = str, default = 'bashes.npy',
        help = 'filename where analysis results will be saved in numpy format')
    args = parser.parse_args()

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(**bashes.great3.Observation.fromArgs(args))

    # Load the postage stamps to analyze.
    dataStamps = obs.getImage()

    # Load the constant PSF stamp to use for the analysis.
    psfStamps = obs.getStars()

    # Load the true noise variance used to simulate this epoch.
    params = obs.getTruthParams()
    noiseVarTruth = params['noise']['variance']

    # Build the estimator for this analysis (using only the first stamp, for now)
    psfModel = obs.createPSF(0)
    estimator = bashes.Estimator(
        data=dataStamps.getStamp(0,0),psfs=psfModel,ivar=1./noiseVarTruth,
        stampSize=obs.stampSize,pixelScale=obs.pixelScale,**bashes.Estimator.fromArgs(args))

    # Analyze stamps using the truth source prior for the first stamp.
    prior = obs.createSource(0)
    priorFlux = prior.getFlux()
    estimator.usePrior(prior,fluxSigmaFraction = 0.1)

    # Save results in numpy format.
    np.save(args.save,estimator.nll)

if __name__ == '__main__':
    main()
