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
    parser.add_argument('--rotate', type = float, default = 90.,
        help = 'Rotation to apply to prior relative to true source (deg)')
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

    # Lookup the catalog truth for this stamp.
    truth = obs.getTruthCatalog()[args.stamp]
    print 'Galaxy SNR =',truth['gal_sn']

    # Center the estimator shear grid on the true shear. Note that this
    # effectively ignores any g1,g2_center values given on the command line.
    # TODO: rework Estimator.addArgs to optionally exclude some args.
    args.g1_center = truth['g1']
    args.g2_center = truth['g2']

    # Build the estimator for this analysis (using only the first stamp, for now)
    estimator = bashes.Estimator(
        data=stamp,psfs=psfModel,ivar=1./noiseVarTruth,
        stampSize=obs.stampSize,pixelScale=obs.pixelScale,**bashes.Estimator.fromArgs(args))

    # Analyze stamps using the truth source prior for the first stamp.
    # Create an unlensed source prior for this stamp using truth info.
    prior = obs.createSource(args.stamp).rotate(args.rotate*galsim.degrees)
    estimator.usePrior(prior,fluxSigmaFraction = 0.1)

    # Find the global minimum NLL wrt (x,y,theta) for the true shear.
    ig,idata = 0,0
    nllMin = np.min(estimator.nllXYTheta[:,ig,idata,:])

    # Initialize matplotlib.
    fig1 = plt.figure('fig1',figsize=(12,9))
    ncol = 5
    nrow = 1+(args.ntheta+ncol-1)//ncol
    xy = estimator.xyGrid
    dxy = xy[1] - xy[0]
    xyEdges = np.linspace(xy[0]-dxy/2,xy[-1]+dxy/2,len(xy)+1)
    xyEdges[0] = xy[0]
    xyEdges[-1] = xy[-1]
    # Lookup the true centroid shift.
    dx = truth['xshift']
    dy = truth['yshift']
    # Initialize contour levels relative to the global minimum NLL.
    nllContours = np.arange(1,11) + nllMin
    # Loop over theta values.
    for ith in range(args.ntheta):
        # Plot NLL(x,y,theta) vs (x,y) at this theta.
        plt.subplot(nrow,ncol,ith+1)
        nll = estimator.nllXYTheta[ith,ig,idata].reshape((args.nxy,args.nxy))
        plt.pcolormesh(xyEdges,xyEdges,nll,cmap='rainbow',rasterized=True)
        # Superimpose contours relative to the global minimum in (x,y,theta).
        plt.contour(xy,xy,nll,levels=nllContours,colors='w',linestyles='-')
        # Draw a marker at the true centroid position.
        plt.plot(dx,dy,marker='x',color='w')
        # Remove tick labels.
        axes = plt.gca()
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
        # Plot NLL(theta) and exp(-NLL(theta)) after (x,y) marginalization.
        ax1 = plt.subplot(nrow,1,nrow)
        nllTheta = estimator.nllTheta[:,ig,idata]
        nllThetaMin = np.min(nllTheta)
        ax1.plot(estimator.thetaGrid,nllTheta - nllThetaMin,'--')
        ax2 = ax1.twinx()
        ax2.yaxis.set_ticklabels([])
        ax2.plot(estimator.thetaGrid,np.exp(-(nllTheta - nllThetaMin)),'-')
    plt.show()

if __name__ == '__main__':
    main()
