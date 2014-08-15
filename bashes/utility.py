import galsim

def render(model,scale,size):
	"""
	Returns a rendering of the specified GalSim model into a postage with dimensions
	size x size with pixel scale scale (in arcsecs). Note that the model will be convolved
	with the pixel response, so should not already be convolved with the pixel response.
	"""
	stamp = galsim.Image(size,size)
	return model.drawImage(image = stamp,scale = scale)
