#!/usr/bin/env python

import argparse
import math

import numpy as np

import galsim
import bashes

def getFeatures(image, psf, stampSize, weight=None):
    
    # fourier transform image
    ftimage = np.fft.fft2(image)
    ftpsf = np.fft.fft2(psf)

    if weight is None:
        weight = np.ones((stampSize,stampSize))
    
    ftdeconvolved = ftimage / ftpsf * weight
        
    # build moment matrix
    kx,ky = np.meshgrid(np.fft.fftfreq(stampSize),np.fft.fftfreq(stampSize))
    kx = kx.flatten()
    ky = ky.flatten()
    kxsq = kx*kx
    kysq = ky*ky
    M = np.array([np.ones(stampSize*stampSize), 1J*kx, 1J*ky, kxsq + kysq, kxsq - kysq, 2*kx*ky])
        
    # return features
    return M.dot(ftdeconvolved.flatten())

def main():

    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    bashes.great3.Observation.addArgs(parser)
    args = parser.parse_args()

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(**bashes.great3.Observation.fromArgs(args))

    # Load the postage stamps to analyze.
    dataStamps = obs.getImage()
    data = dataStamps.getStamp(0,0)

    # Load the constant PSF stamp to use for the analysis.
    psfStamps = obs.getStars()
    psf = psfStamps.getStamp(0,0)

    features = getFeatures(data.array,psf.array,stampSize=obs.stampSize)

    print features


if __name__ == '__main__':
    main()
