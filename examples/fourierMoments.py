#!/usr/bin/env python

import argparse
import math

import numpy as np

import galsim
import bashes

def getFeatures(image, psfModel, sigma):
    stampSize = image.xmax - image.xmin + 1
    pixelScale = image.scale

    # fourier transform image
    ftimage = np.fft.fft2(image.array)

    # render psf and fourier transform
    psf = bashes.utility.render(psfModel,scale=pixelScale,size=stampSize)
    ftpsf = np.fft.fft2(psf.array)
        
    # build weight function
    kx,ky = np.meshgrid(np.fft.fftfreq(stampSize),np.fft.fftfreq(stampSize))
    sigmasqby2 = sigma*sigma/2
    weight = np.exp(-kx*ky*sigmasqby2)    
    
    # deconvolve image
    ftdeconvolved = ftimage / ftpsf * weight

    # build moment matrix
    kxsq = kx*kx
    kysq = ky*ky
    moments = [np.ones((stampSize,stampSize)), 1J*kx, 1J*ky, kxsq + kysq, kxsq - kysq, 2*kx*ky]
    M = np.array([x.flatten() for x in moments])
        
    # return features
    return M.dot(ftdeconvolved.flatten())

def main():

    # Parse command-line args.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sigma', type = float, default = 1.,
            help = 'Weight function size')
    bashes.great3.Observation.addArgs(parser)
    args = parser.parse_args()

    # Initialize the GREAT3 observation we will analyze.
    obs = bashes.great3.Observation(**bashes.great3.Observation.fromArgs(args))

    # Load the postage stamps to analyze.
    dataStamps = obs.getImage()
    data = dataStamps.getStamp(0,0)

    # Load the constant PSF stamp to use for the analysis.    
    psfModel = obs.createPSF(0)

    features = getFeatures(image=data,psfModel=psfModel,sigma=args.sigma)

    print features


if __name__ == '__main__':
    main()
