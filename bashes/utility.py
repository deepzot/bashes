import galsim

def render(model,scale,size=None,stamp=None):
	"""
	Returns a rendering of the specified GalSim model into a square postage stamp with the
	specified pixel scale (arcsecs). Either a stamp pixel size or an existing stamp (with its
	scale attribute set) must be provided. Note that the model will be convolved with the pixel
	response, so should not already be convolved with the pixel response.
	"""
	if stamp is None:
		stamp = galsim.Image(size,size)
		stamp.scale = scale
	return model.drawImage(image = stamp)
