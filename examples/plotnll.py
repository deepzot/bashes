#!/usr/bin/env python

# Creates plots to illustrate the marginalization algorithm.

import argparse

import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import galsim
import bashes

def main():

    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bashes.great3.Observation.addArgs(parser)
    parser.add_argument('--data-index', type = int, default = 0,
        help = 'Index of data stamp to analyze (0-9999')
    parser.add_argument('--prior-index', type = int, default = None,
        help = 'Index of prior to assume (0-9999, default is to use data index)')
    parser.add_argument('--rotate', type = float, default = 90.,
        help = 'Rotation to apply to prior relative to true source (deg)')
    parser.add_argument('--dg1', type = float, default = 0.,
        help = 'Amount to offset g1 component of estimator shear from true value')
    parser.add_argument('--dg2', type = float, default = 0.,
        help = 'Amount to offset g2 component of estimator shear from true value')
    parser.add_argument('--save', type = str, default = None,
        help = 'base filename for saving results')
    bashes.Estimator.addArgs(parser)
    args = parser.parse_args()

    # Check stamp index args.
    if args.data_index < 0 or args.data_index >= 10000:
        print 'data index %d is outside of valid range 0-10000' % args.data_index
        return -1
    if args.prior_index is not None and (args.prior_index < 0 or args.prior_index >= 10000):
        print 'prior index %d is outside of valid range 0-10000' % args.prior_index
        return -1

    # Save our config if requested.
    if args.save:
        config = bashes.config.Config(args)
        config.save(args.save)

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(**bashes.great3.Observation.fromArgs(args))

    # Load the one simulated postage stamp we will use to generate plots.
    stamp = obs.getImage().getStamp(args.data_index)

    # Create the PSF model for this stamp using truth info.
    psfModels = [ obs.createPSF(args.data_index) ]

    # Load the true noise variance used to simulate this epoch.
    params = obs.getTruthParams()
    noiseVarTruth = params['noise']['variance']

    # Lookup the catalog truth for this stamp.
    truth = obs.getTruthCatalog()[args.data_index]
    print 'Galaxy SNR =',truth['gal_sn']

    # Estimator uses a single shear value given by the true shear plus some offset.
    args.nshear = 1
    args.g1_center = truth['g1'] + args.dg1
    args.g2_center = truth['g2'] + args.dg2

    # Build the estimator for this analysis (using only the first stamp, for now)
    estimator = bashes.Estimator(
        data=stamp,psfs=psfModels,ivar=1./noiseVarTruth,
        stampSize=obs.stampSize,pixelScale=obs.pixelScale,**bashes.Estimator.fromArgs(args))

    # Select the prior to use.
    if args.prior_index is None:
        args.prior_index = args.data_index
    priorModel = obs.createSource(args.prior_index).rotate(args.rotate*galsim.degrees)

    # Analyze stamps using the truth source prior for the first stamp.
    estimator.usePrior(priorModel,fluxSigmaFraction = 0.1)

    # Find the global minimum NLL wrt (x,y,theta) for the true shear.
    ig,idata = 0,0
    nllMin = np.min(estimator.nllXYTheta[:,ig,idata,:])

    # Initialize matplotlib.
    fig = plt.figure('fig1',figsize=(12,9))
    fig.set_facecolor('white')
    plt.subplots_adjust(left=0.05,bottom=0.06,right=0.98,top=0.98,wspace=0.1,hspace=0.1)
    ncol = 5
    nrow = 1+(args.ntheta+ncol-1)//ncol
    xy = estimator.xyFine
    dxy = xy[1] - xy[0]
    xyEdges = np.linspace(xy[0]-dxy/2,xy[-1]+dxy/2,len(xy)+1)
    coarseGrid = estimator.xyGrid[1:-1]
    # Lookup the true centroid shift.
    dx = truth['xshift']
    dy = truth['yshift']
    # Initialize contour levels relative to the global minimum NLL.
    nllProb = np.array((0.6827,0.9543,0.9973))
    nllDeltaChi2 = scipy.stats.chi2.isf(1-nllProb,df=3)
    # Loop over theta values.
    for ith in range(args.ntheta):
        # Plot NLL(x,y,theta) vs (x,y) at this theta.
        plt.subplot(nrow,ncol,ith+1)
        nll = estimator.getNllXYFine(ith,ig,idata)
        plt.pcolormesh(xyEdges,xyEdges,nll,cmap='rainbow',rasterized=True)
        # Superimpose contours relative to the global minimum in (x,y,theta).
        plt.contour(xy,xy,nll,levels=nllMin+nllDeltaChi2,
            colors='w',linestyles=('-','--',':'))
        # Remove tick labels.
        axes = plt.gca()
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
        # Draw tick marks to show the coarse grid.
        axes.set_xticks(coarseGrid)
        axes.set_yticks(coarseGrid)
        # Draw a marker at the true (x,y) centroid location.
        plt.plot(dx,dy,'r*',markersize=15)
    # Plot NLL(theta) after (x,y) marginalization.
    ax1 = plt.subplot(nrow,1,nrow)
    nllTheta = estimator.getNllFine(ig,idata)
    nllThetaMin = np.min(nllTheta)
    print 'min(NLL) =',nllThetaMin
    dnllTheta = nllTheta - nllThetaMin
    ax1.plot(estimator.thetaFine,dnllTheta,'b--')
    ax1.plot(estimator.thetaGrid,
        estimator.nllTheta[:,ig,idata]-nllThetaMin,'k.',markersize=10)
    yrange = np.max(dnllTheta)
    plt.ylim((-0.05*yrange,1.05*yrange))
    plt.xlabel('Source rotation (deg)')
    plt.ylabel('NLL - min(NLL)')
    # Superimpose the likelihood exp(-NLL(theta)).
    ax2 = ax1.twinx()
    ax2.yaxis.set_ticklabels([])
    ax2.plot(estimator.thetaFine,np.exp(-dnllTheta),'b-')
    # Draw a marker at the true value.
    plt.ylim((0.,1.05))
    ax2.plot(args.rotate,0.15,'r*',markersize=15)
    plt.show()

if __name__ == '__main__':
    main()
