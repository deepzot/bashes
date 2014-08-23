# SConstruct is python -*-Python-*-
import os

# initialize our build enviroment
env = Environment(CPPPATH=['#'],LIBPATH=[])

# take PATH and LD_LIBRARY_PATH from the environment so currently configured
# build tools are used
if 'PATH' in os.environ:
   env['ENV']['PATH'] = os.environ['PATH']
if 'LD_LIBRARY_PATH' in os.environ:
   env['ENV']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']

# take CC and CXX from the current environment if they are defined
if 'CC' in os.environ:
   env['CC'] = os.environ['CC']
   print 'Using CC =',env['CC']
if 'CXX' in os.environ:
   env['CXX'] = os.environ['CXX']
   print 'Using CXX =',env['CXX']

# configure the environment to find the packages we need
conf = Configure(env)

# copy any library paths defined in $LDFLAGS to LIBPATH
if 'LDFLAGS' in os.environ:
   for token in os.environ['LDFLAGS'].split():
      if token[:2] == '-L':
         conf.env.Append(LIBPATH=[token[2:]])

# take include,lib paths from environment variables 
def takeFromEnv(incName,libName):
   if incName and incName in os.environ:
      conf.env.AppendUnique(CPPPATH=[os.environ[incName]])
   if libName and libName in os.environ:
      conf.env.AppendUnique(LIBPATH=[os.environ[libName]])

'''
# zlib (needed by hdf5)
takeFromEnv('','ZLIBLIBDIR')
if not conf.CheckLib('z'):
   Exit(1)
# hdf5
takeFromEnv('HDF5INCLUDEDIR','HDF5LIBDIR')
if not conf.CheckCHeader('hdf5.h') or not conf.CheckLib('hdf5'):
   Exit(1)
# fftw3
takeFromEnv('FFTW3INCLUDEDIR','FFTW3LIBDIR')
if not conf.CheckCHeader('fftw3.h') or not conf.CheckLib('fftw3'):
   Exit(1)
# cfitsio
takeFromEnv('CFITSIOINCLUDEDIR','CFITSIOLIBDIR')
if not conf.CheckCHeader('fitsio.h') or not conf.CheckLib('cfitsio'):
   Exit(1)
# tmv
takeFromEnv('TMVINCLUDEDIR','TMVLIBDIR')
if not conf.CheckLib('tmv',language='C++'):
   Exit(1)
# galsim
takeFromEnv('GALSIMINCLUDEDIR','GALSIMLIBDIR')
if not conf.CheckCXXHeader('GalSim.h') or not conf.CheckLib('galsim',language='C++'):
   Exit(1)
'''
# boost
takeFromEnv('BOOST_INC_DIR','BOOST_LIB_DIR')
if not conf.CheckCXXHeader('boost/system/error_code.hpp'):
   Exit(1)
# all done: update the build environment
env = conf.Finish()

# build C++ library
SConscript('src/SConscript', exports='env', variant_dir='build')

# build C++ programs
penv = env.Clone()
penv.AppendUnique(LIBS=['bashes','boost_program_options','boost_thread','boost_system'])
penv.AppendUnique(LIBPATH=['build'])
penv.AppendUnique(RPATH=['build'])
#penv.Program('scan','scan.cc')
#penv.Program('gpuinit','gpuinit.cc')
#penv.Program('tpool','tpool.cc')

# build CUDA programs if nvcc is in our path
if not penv.WhereIs('nvcc'):
   print 'Skipping CUDA build phase since nvcc not found.'
else:
   cenv = penv.Clone()
   if 'CUDA_ROOT' in os.environ:
      cudaRoot = os.environ['CUDA_ROOT']
      cenv.Append(LIBPATH=[cudaRoot+'/lib64'])

   # We perform separate compilation and linking below as described at
   # http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#using-separate-compilation-in-cuda
   '''
   # Compile the mixed CPU, GPU code. There are a few hacks below:
   #  - fftw thinks nvcc is gcc/4.8.2 and expects to have __float128 defined as a quad float type
   #  - https://github.com/milakov/nnForge/issues/1
   #  - missing some headers that define __locale_t, etc
   # Even with these hacks, boost/smart_ptr generates some warnings about "cc" clobber and variable "tmp".
   gputest = cenv.Command('gputest.o','gputest.cu',"nvcc -D__float128=double -DBOOST_NOINLINE='__attribute__ ((noinline))' -D__USE_XOPEN2K8 $CXXFLAGS $_CPPINCFLAGS -arch sm_35 -m64 -dc $SOURCE -o $TARGET")
   gpubash = cenv.Command('gpubash.o','gpubash.cu',"nvcc -Xptxas -v -D__float128=double -DBOOST_NOINLINE='__attribute__ ((noinline))' -D__USE_XOPEN2K8 $CXXFLAGS $_CPPINCFLAGS -arch sm_35 -m64 -dc $SOURCE -o $TARGET")
   Depends([gputest,gpubash],'kernels.h')
   Depends([gpubash],['spline5x5.h','spline7x7.h'])

   # link the GPU code
   cenv.Append(LIBS=['cudart','cudadevrt','cublas','m','stdc++'])
   cenv.Command('gputest_link.o','gputest.o',"nvcc -arch sm_35 -m64 -dlink $SOURCE -o $TARGET $_LIBPATHFLAGS $_LIBFLAGS")
   cenv.Command('gpubash_link.o','gpubash.o',"nvcc -arch sm_35 -m64 -dlink $SOURCE -o $TARGET $_LIBPATHFLAGS $_LIBFLAGS")

   # link the CPU code
   cenv.Program('gputest',['gputest.o','gputest_link.o'])
   cenv.Program('gpubash',['gpubash.o','gpubash_link.o'])
   '''
