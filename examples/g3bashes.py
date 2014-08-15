#!/usr/bin/env python

import os
import os.path
import argparse

from astropy.io import fits
import yaml

import bashes

def main():

    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--branch', type = str, default = 'control/ground/constant',
        help = 'Name of branch to use relative to $GREAT3_ROOT')
    parser.add_argument('--field', type = int, default = 0,
        help = 'Index of field to analyze (0-199)')
    parser.add_argument('--epoch', type = int, default = 0,
        help = 'Epoch number to analyze')
    args = parser.parse_args()

    # Lookup the path to the GREAT3 branch we will use.
    if 'GREAT3_ROOT' not in os.environ:
        print '$GREAT3_ROOT is not set.'
        return -1
    branchPath = os.path.join(os.environ['GREAT3_ROOT'],args.branch)
    if not os.path.isdir(branchPath):
        print 'No such directory:',branchPath
        return -1

    # Load the postage stamps to analyze.
    if args.field < 0 or args.field >= 200:
        print 'Field index %d out of range 0-199.' % args.field
        return -1
    dataStampsPath = os.path.join(branchPath,'image-%03d-0.fits' % args.field)
    hduList = fits.open(dataStampsPath)
    dataStamps = hduList[0].data
    hduList.close()

    # Infer the stamp size from the stamp array shape.
    assert dataStamps.shape[0] == dataStamps.shape[1], 'Image data is not square'
    assert dataStamps.shape[0] % 100 == 0, 'Image data does not consist of 100x100 stamps'
    stampSize = dataStamps.shape[0]//100

    # Load the constant PSF stamp to use for the analysis.
    constPsfPath = os.path.join(branchPath,'starfield_image-%03d-0.fits' % args.field)
    hduList = fits.open(constPsfPath)
    constPsf = hduList[0].data[:stampSize,:stampSize]
    hduList.close()

    # Load the simulation truth for this data.
    truthParamsPath = os.path.join(os.environ['GREAT3_ROOT'],'truth',args.branch,
        'epoch_parameters-%03d-%d.yaml' % (args.field,args.epoch))
    with open(truthParamsPath,'r') as f:
        params = yaml.load(f)
        noiseVar = params['noise']['variance']

    # Build the estimator for this analysis.
    estimator = bashes.Estimator(
        data=dataStamps,psfs=constPsf,ivar=1./noiseVar,
        stampSize=stampSize)

if __name__ == '__main__':
    main()
