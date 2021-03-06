#!/usr/bin/env python

import argparse
import os.path

import numpy as np
import matplotlib.pyplot as plt

import galsim
import bashes

def main():

    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--load', type = str, default = 'bashes',
        help = 'base filename of g3bashes job we will post process')
    parser.add_argument('--grid', type = int, default = 0,
        help = 'number of stamp/prior combinations to show as a grid of plots')
    parser.add_argument('--verbose', action = 'store_true',
        help = 'be verbose about progress')
    parser.add_argument('--save', type = str, default = None,
        help = 'base name for saving ouput')
    args = parser.parse_args()

    # Load the config data for the g3bashes job we are post processing.
    config = bashes.config.load(args.load)
    saveBase = config['args']['save']

    # Reconstruct the shear grid used by the estimator.
    ng = config['args']['ng']
    gmax = config['args']['gmax']
    dg = np.linspace(-gmax,+gmax,ng)
    g1 = config['args']['g1_center'] + dg
    g2 = config['args']['g2_center'] + dg

    # Prepare the shear grid edges needed by pcolormesh.
    g1e,g2e = bashes.utility.getBinEdges(g1,g2)

    # Initialize matplotlib.
    if args.grid:
        gridFig = plt.figure('fig1',figsize=(12,12))
        gridFig.set_facecolor('white')
        plt.subplots_adjust(left=0.02,bottom=0.02,right=0.98,top=0.98,wspace=0.05,hspace=0.05)

    # Allocate memory for the full NLL grid over all priors.
    nstamps = config['args']['nstamps']
    nll = np.empty((ng*ng,nstamps,nstamps))

    # Load this array from the NLL grids saved for each prior.
    for iprior in range(nstamps):
        # Load the estimator results for this prior.
        loadName = '%s_%d.npy' % (saveBase,iprior)
        if not os.path.exists(loadName):
            print 'Skipping missing results for prior %d in %r' % (i,loadName)
            continue
        nll[:,:,iprior] = np.load(loadName)

    # Marginalize over priors for each data stamp.
    nllData = np.empty((ng*ng,nstamps))
    for idata in range(nstamps):
        for ig in range(ng*ng):
            nllData[ig,idata] = bashes.Estimator.marginalize(nll[ig,idata])

    # Sum the NLL over data stamps, assuming that the same constant shear is applied.
    nllTotal = np.sum(nllData,axis=1)
    nllTotalMin = np.min(nllTotal)

    # Define a shear plot helper.
    nllLevels = bashes.utility.getDeltaChiSq()
    def plotShearNLL(nll):
        nllShear = nll.reshape((ng,ng))
        nllShearMin = np.min(nllShear)
        plt.pcolormesh(g1e,g2e,nllShear,cmap='rainbow',vmin=0.,vmax=nllLevels[-1])
        plt.contour(g1,g2,nllShear,levels=nllLevels,colors='w',linestyles=('-','--',':'))
        # Remove tick labels.
        axes = plt.gca()
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])

    # Draw a grid of shear NLL values if requested.
    if args.grid:
        # Show the shear grid for each stamp,prior pair.
        for iprior in range(args.grid):
            for idata in range(args.grid):
                plt.subplot(args.grid+1,args.grid+1,iprior*(args.grid+1)+idata+1)
                plotShearNLL(nll[:,idata,iprior]-np.min(nllData[:,idata]))
        # Show the shear grid marginalized over priors for each data stamp.
        for idata in range(args.grid):
            plt.subplot(args.grid+1,args.grid+1,args.grid*(args.grid+1)+idata+1)
            plotShearNLL(nllData[:,idata]-np.min(nllData[:,idata]))
        # Show the combined NLL assuming constant shear.
        plt.subplot(args.grid+1,args.grid+1,(args.grid+1)**2)
        plotShearNLL(nllTotal-nllTotalMin)
        if args.save:
            plt.savefig(args.save)
        plt.show()

if __name__ == '__main__':
    main()
