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

    # Loop over priors.
    for i in range(config['args']['nstamps']):
        # Load the estimator results for this prior.
        loadName = '%s_%d.npy' % (saveBase,i)
        if not os.path.exists(loadName):
            print 'Skipping missing results for prior %d in %r' % (i,loadName)
            continue
        nll = np.load(loadName)
        print nll.shape

if __name__ == '__main__':
    main()
