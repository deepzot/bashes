"""
Standardized handling of config info
"""

import sys
import time
import os
import os.path
import subprocess
import yaml

def githash(what='HEAD'):
    """
    Returns the hash for the HEAD of the git repo containing this file or a helpful
    error message string if this cannot be determined
    """
    # find the path of this file
    mypath = os.path.dirname(os.path.realpath(__file__))
    # get our git repo path assuming a standard checkout
    gitpath = os.path.join(mypath,os.pardir,'.git')
    if os.path.exists(gitpath):
        try:
            hash = subprocess.check_output(['git','--git-dir',gitpath,'rev-parse',what],
            	stderr=subprocess.STDOUT).rstrip()
        except subprocess.CalledProcessError,e:
            hash = e.output
        except Exception,e:
            hash = repr(e)
    else:
        hash = 'cannot find git repo for %r' % mypath
    return hash

class Config(dict):
    """
    Represents a dictionary of serializable configuration data.
    """
    def __init__(self,args=None):
        """
        Creates and returns a new dictionary with some standard headers to identify the
        process that created this object. The optional args should be a dictionary (or
        something convertible to a dictionary via vars()) of parsed command-line args.
        """
        self['time'] = time.ctime()
        self['argv'] = sys.argv
        self['uname'] = list(os.uname())
        self['git'] = githash()
        if args is not None:
            # vars() will convert an argparse.Namespace(...) into an arg dictionary
            self['args'] = vars(args)

    def save(self,name,noclobber=False):
        """
        Serializes this config to a YAML file using the specified base name. Will not overwrite
        an existing file if noclobber is True. The .yaml extension is added to name unless it
        already has an extension.
        """
        # Add a .yaml extension if no extension is already present in name.
        base,ext = os.path.splitext(name)
        if not ext:
        	name += '.yaml'
        # Can we clobber an existing file?
        if os.path.exists(name) and noclobber:
            raise RuntimeError('Will not clobber %r' % name)
        with open(name,'w') as out:
            yaml.dump(dict(self),stream = out)

def load(name):
    """
    Loads and returns a dictionary for the named config. The .yaml extension is added to name
    unless it already has an extension. Note that the return value is a plain dict and not a
    Config object.
    """
    # Add a .yaml extension if no extension is already present in name.
    base,ext = os.path.splitext(name)
    if not ext:
    	name += '.yaml'
    if not os.path.exists(name):
        raise RuntimeError('Config not found: %r' % name)
    with open(name,'r') as fin:
        results = yaml.load(stream = fin)
    return results
