"""
Supports interactive display to a separate DS9 process via XPA. For documentation of XPA
see http://hea-www.harvard.edu/saord/ds9/ref/xpa.html
"""

import numpy as np

class Display(object):
    """
    Represents a connection with a running DS9 process.
    """
    def __init__(self,config=None):
        """
        Initializes a new connection. This will normally start DS9 running, if necessary,
        or else connect to a previously running instance of DS9. Any config commands provided
        in the optional cmds string will be sent. Separate multiple commands with semicolons.
        """
        import ds9
        self.display = ds9.ds9(start='-geometry 900x1000')
        # are we connecting to a previously running session?
        self.frames = int(self.get('frame frameno'))
        if self.frames == 1 and self.get('frame has fits') == 'no':
            self.frames = 0
        # set defaults for future frames
        self.config(config)

    def set(self,cmd):
        return self.display.set(cmd)
    def get(self,cmd):
        return self.display.get(cmd)

    def config(self,cmds):
        """
        Sends commands separated by semicolons
        """
        if cmds is not None:
            for cmd in cmds.split(';'):
                cmd = cmd.strip()
                if cmd:
                    self.display.set(cmd)

    def show(self,image,min=None,max=None,config=None,reuseLimits=False):
        """
        Displays the specified image in a new frame. The image can be provided as
        a numpy array or a galsim Image type. Scale limits will
        be set automatically from the data unless min or max are specified. If
        reuseLimits is True and this is not the first frame, then min/max are ignored
        and the limits of the current frame are reused for this image.
        Any config commands provided in the optional cmds string will be
        sent. Separate multiple commands with semicolons. By default the frame
        is zoomed to fit, but this can be overridden in cmds.
        """
        # detect galsim image types by the presence of an array attribute
        if hasattr(image,'array'):
            image = image.array
        # determine the scale limits we will use
        if reuseLimits and self.frames > 0:
            limits = self.get('scale limits')
        else:
            if min is None:
                min = np.min(image)
            if max is None:
                max = np.max(image)
            limits = "%g %g" % (min,max)
        # create a new frame unless this is the first time we have been called
        if self.frames > 0:
            self.display.set('frame new')
        # display the numpy array
        self.display.set_np2arr(image.astype(np.float32,copy=False))
        self.display.set('scale limits %s' % limits)
        # zoom to fit
        self.display.set('zoom to fit')
        # send any additional commands to configure this frame
        self.config(config)
        self.frames += 1
