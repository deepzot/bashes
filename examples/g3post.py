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

    # Loop over priors.
    nstamps = config['args']['nstamps']
    for iprior in range(nstamps):
        # Load the estimator results for this prior.
        loadName = '%s_%d.npy' % (saveBase,iprior)
        if not os.path.exists(loadName):
            print 'Skipping missing results for prior %d in %r' % (i,loadName)
            continue
        nll = np.load(loadName)
        nllMin = np.min(nll)
        nllLevels = bashes.utility.getDeltaChiSq()
        assert nll.shape[0] == ng*ng
        ndata = nll.shape[1]
        for idata in range(ndata):
            nllShear = nll[:,idata].reshape((ng,ng))-nllMin
            print 'prior %d, stamp %d, nllMin = %f (%f)' % (iprior,idata,np.min(nllShear),nllMin)
            if args.grid and iprior < args.grid and idata < args.grid:
                plt.subplot(args.grid,args.grid,iprior*args.grid+idata+1)
                plt.pcolormesh(g1e,g2e,nllShear,cmap='rainbow')
                plt.contour(g1,g2,nllShear,levels=nllLevels,colors='w',linestyles=('-','--',':'))
                # Remove tick labels.
                axes = plt.gca()
                axes.xaxis.set_ticklabels([])
                axes.yaxis.set_ticklabels([])
    plt.show()

if __name__ == '__main__':
    main()
