import sys
import os
import time
import importlib
from pathlib import Path

# Path to a separate development version of numpy with fixes
DEV_NUMPY = '~/sources/numpy/'

# a little bit of a hack to import different versions of NumPy
fft = sys.argv[1]
if 'fixed' in fft:
    sys.path.insert(0, DEV_NUMPY)

# other imports (which could possibly import numpy)
import h5py
import config
import numpy as np
import scipy.fftpack
import pyfftw
from lib import l2error, l1error, linferror, scipy_rfft

np.random.seed(config.SPEED_SEED)

results = {fft: {'sizes': [], 'complex': [], 'real': [],
           'complex_init': [], 'real_init': []}}


def numpy_clear_cache(x):
    # hacky but seems to work...
    from numpy.fft import fftpack
    n = x.shape[0]
    while fftpack._fft_cache.pop_twiddle_factors(n) is not None:
        pass
    while fftpack._real_fft_cache.pop_twiddle_factors(n) is not None:
        pass


def scipy_clear_cache(x):
    import scipy.fftpack._fftpack as sff
    sff.destroy_zfft_cache()
    sff.destroy_zfftnd_cache()
    sff.destroy_drfft_cache()
    sff.destroy_cfft_cache()
    sff.destroy_cfftnd_cache()
    sff.destroy_rfft_cache()
    sff.destroy_ddct2_cache()
    sff.destroy_ddct1_cache()
    sff.destroy_dct2_cache()
    sff.destroy_dct1_cache()
    sff.destroy_ddst2_cache()
    sff.destroy_ddst1_cache()
    sff.destroy_dst2_cache()
    sff.destroy_dst1_cache()


def time_operation(fun, repeat, *args, setup_call=None):
    ''' Repeats the fft operation and returns the _minimal_ runtime.
    '''
    times = []
    if setup_call is None:
        # first time for building cache
        fun(*args)
    for i in range(10):
        if setup_call is not None:
            setup_call(*args)
        start = time.perf_counter()
        fun(*args)
        stop = time.perf_counter()
        times.append(stop - start)
    return np.min(times)

N2 = config.SPEED_REPEAT
for N in config.SPEED_SIZES:
    ca = (np.random.uniform(-0.5, 0.5, N) +
          1.0j * np.random.uniform(-0.5, 0.5, N))
    ra = np.random.uniform(-0.5, 0.5, N)
    if 'pyfftw' in fft:
        ca = pyfftw.byte_align(ca)
        ra = pyfftw.byte_align(ra)
        ra2 = pyfftw.empty_aligned(ra.shape[0]//2+1, np.complex128)
        ca2 = pyfftw.empty_aligned(ca.shape, ca.dtype)
        initfun = lambda a, b: pyfftw.FFTW(a, b, flags=('FFTW_ESTIMATE',))
        init_time_real = time_operation(initfun, N2, ra, ra2)
        init_time_complex = time_operation(initfun, N2, ca, ca2)
        cfft = pyfftw.FFTW(ca, ca2, flags=('FFTW_ESTIMATE',))
        rfft = pyfftw.FFTW(ra, ra2, flags=('FFTW_ESTIMATE',))
        time_real = time_operation(cfft.execute, N2)
        time_complex = time_operation(rfft.execute, N2)
    else:
        if 'numpy' in fft:
            rfftfun = np.fft.rfft
            fftfun = np.fft.fft
            cc = numpy_clear_cache
        elif fft == 'scipy':
            rfftfun = scipy.fftpack.rfft
            fftfun = scipy.fftpack.fft
            cc = scipy_clear_cache
        # measure the init times (with creation of twiddle factors)
        init_time_real = time_operation(rfftfun, N2, ra, setup_call=cc)
        init_time_complex = time_operation(fftfun, N2, ca, setup_call=cc)
        # measure the calculation time
        time_real = time_operation(rfftfun, N2, ra)
        time_complex = time_operation(fftfun, N2, ca)
    results[fft]['sizes'].append(N)
    results[fft]['complex'].append(time_complex)
    results[fft]['complex_init'].append(init_time_complex)
    results[fft]['real'].append(time_real)
    results[fft]['real_init'].append(init_time_real)

# save results
for k in results:
    res = results[k]
    with h5py.File(config.SPEED_RESULT_DIR + '/' + k + '.hdf5', 'w') as f:
        for k2 in res:
            res[k2] = np.array(res[k2])
        sidx = np.argsort(res['sizes'])
        for k2 in res:
            f.create_dataset(k2, data=res[k2][sidx])
