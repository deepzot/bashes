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
    parser.add_argument('--sigma-frac', type = float, default = 0.1,
        help = 'fraction of flux prior to use as flux sigma')
    parser.add_argument('--nstamps', type = int, default = 0,
        help = 'number of stamps to include in the analysis (or 0 to use all stamps)')
    parser.add_argument('--save', type = str, default = 'bashes',
        help = 'base filename used to save estimator results for each prior')
    parser.add_argument('--verbose', action = 'store_true',
        help = 'be verbose about progress')
    args = parser.parse_args()

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(**bashes.great3.Observation.fromArgs(args))

    # Load the postage stamps to analyze and their corresponding PSF models.
    image = obs.getImage()
    nstamps = args.nstamps if args.nstamps > 0 else image.nx*image.ny
    dataStamps = np.empty((nstamps,obs.stampSize,obs.stampSize))
    psfModels = [ ]
    for i in range(nstamps):
        dataStamps[i] = image.getStamp(i).array
        psfModels.append(obs.createPSF(i))

    # Load the true noise variance used to simulate this epoch.
    params = obs.getTruthParams()
    noiseVarTruth = params['noise']['variance']

    # Save our config if requested.
    if args.save:
        config = bashes.config.Config(args)
        config.save(args.save)

    # Build the estimator for this analysis (using only the first stamp, for now)
    estimator = bashes.Estimator(
        data=dataStamps,psfs=psfModels,ivar=1./noiseVarTruth,
        stampSize=obs.stampSize,pixelScale=obs.pixelScale,**bashes.Estimator.fromArgs(args))

    # Analyze stamps using the truth source priors for each stamp.
    traceMsg = None
    for i in range(nstamps):
        prior = obs.createSource(i)
        if args.verbose:
            traceMsg = 'prior %d/%d, theta %%d/%d, shear %%d/%d' % (i,nstamps,args.ntheta,args.ng**2)
        estimator.usePrior(prior,fluxSigmaFraction = args.sigma_frac,traceMsg = traceMsg)
        # Save results for this prior in numpy format.
        saveFile = '%s_%d.npy' % (args.save,i)
        np.save(saveFile,estimator.nll)
        if args.verbose:
            print 'Saved estimator results for prior %d to %s' % (i,saveFile)

if __name__ == '__main__':
    main()
