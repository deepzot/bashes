#!/usr/bin/env python

import argparse
import math

import numpy as np

import galsim
import bashes

import scipy.ndimage
import scipy.interpolate

def getTSquare(psf):
    # fourier transform and shift
    t = np.fft.fftshift(np.fft.fft2(psf.array))
    tSq = (np.conjugate(t)*t).real
    # average over theta
    sx, sy = tSq.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(X - sx/2 + 0.5, Y - sy/2 + 0.5)
    rbin = r.astype(np.int)
    tSqAvg = scipy.ndimage.mean(tSq, labels=rbin, index=np.arange(0, rbin.max()+1))
    # build 2d representation
    tSqAvgInterp = scipy.interpolate.InterpolatedUnivariateSpline(np.arange(len(tSqAvg)),tSqAvg)
    tSq2d = tSqAvgInterp(r.flatten()).reshape(sx,sy)
    # shift and return
    return np.fft.fftshift(tSq2d)

def getFeatures(image, psfModel, sigma):
    """
    Returns Fourier moment features of the provided image.
    """
    # Get stamp size/scale
    stampSize = image.xmax - image.xmin + 1
    pixelScale = image.scale

    # Fourier transform image
    ftimage = np.fft.fft2(image.array)

    # Render psf and fourier transform
    psf = bashes.utility.render(psfModel,scale=pixelScale,size=stampSize)
    ftpsf = np.fft.fft2(psf.array)
        
    # k grid
    kx,ky = np.meshgrid(np.fft.fftfreq(stampSize),np.fft.fftfreq(stampSize))
    kxsq = kx*kx
    kysq = ky*ky
    ksq = kxsq + kysq

    # Build weight function
    sigmasqby2 = sigma*sigma/2
    wg = np.exp(-ksq*sigmasqby2)
    tsq = getTSquare(psf)
    weight = wg*tsq
    
    # Deconvolve image
    ftdeconvolved = ftimage / ftpsf * weight

    # Build moment matrix
    moments = [np.ones((stampSize,stampSize)), 1J*kx, 1J*ky, ksq, kxsq - kysq, 2*kx*ky]
    M = np.array([x.flatten() for x in moments])
        
    # Return features
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
