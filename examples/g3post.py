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
    parser.add_argument('--verbose', action = 'store_true',
        help = 'be verbose about progress')
    args = parser.parse_args()

    # Load the config data for the g3bashes job we are post processing.
    config = bashes.config.load(args.load)
    saveBase = config['args']['save']

    # Reconstruct the shear grid used by the estimator.
    ng = config['args']['ng']
    gmax = config['args']['gmax']
    g1_center = config['args']['g1_center']
    g2_center = config['args']['g2_center']
    dg = np.linspace(-gmax,+gmax,ng)
    g1 = g1_center+dg
    g2 = g2_center+dg

    # Prepare the shear grid edges needed by pcolormesh.
    g1e,g2e = bashes.utility.getBinEdges(g1,g2)

    # Initialize matplotlib.
    fig = plt.figure('fig1',figsize=(12,9))
    fig.set_facecolor('white')
    #plt.subplots_adjust(left=0.05,bottom=0.06,right=0.98,top=0.98,wspace=0.1,hspace=0.1)

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
        ndata = nll.shape[0]
        for idata in range(nstamps):
            print 'prior %d, stamp %d' % (iprior,idata)
            plt.pcolormesh(g1e,g2e,nll[:,idata].reshape((ng,ng))-nllMin,cmap='rainbow')
            plt.show()

if __name__ == '__main__':
    main()
