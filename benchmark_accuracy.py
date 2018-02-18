import sys
import os
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
from lib import l2error, l1error, linferror, scipy_rfft

# The routines are stored in a tuple
# 1 - forward complex FFT - numpy convention
# 2 - identity complex FFT - e.g. ifft(fft(x))
# 3 - forward real FFT
# 4 - identity real FFT - e.g. irfft(rfft(x))
if 'numpy' in fft:
    routines = (np.fft.fft, lambda x: np.fft.ifft(np.fft.fft(x)),
                np.fft.rfft, lambda x: np.fft.irfft(np.fft.rfft(x)))
elif fft == 'scipy':
    routines = (scipy.fftpack.fft,
                lambda x: scipy.fftpack.ifft(scipy.fftpack.fft(x)),
                scipy_rfft,
                lambda x: scipy.fftpack.irfft(scipy.fftpack.rfft(x)))
elif fft == 'pyfftw':
    import pyfftw.interfaces.numpy_fft as pyfftw
    routines = (pyfftw.fft, lambda x: pyfftw.ifft(pyfftw.fft(x)),
                pyfftw.rfft, lambda x: pyfftw.irfft(pyfftw.rfft(x)))

ffts = {fft: routines}
results = {k: {'sizes': [], 'complex_l2': [], 'complex_l1': [],
               'complex_linf': [], 'real_l2': [], 'real_l1': [],
               'real_linf': [], 'identity': [], 'real_identity': [],
               'identity_sizes': []} for k in ffts}

# calculate the forward error to a high-precision FFT calculation
for p in sorted(Path(config.DATA_DIR).iterdir()):
    if not p.is_file() or 'self' in p.stem:
        continue
    with p.open('rb') as f:
        ca = np.load(f)
        exact_ft_ca = np.load(f)
        ra = np.load(f)
        exact_ft_ra = np.load(f)
    N = ca.shape[0]
    for k in ffts:
        cfft, cidentity, rfft, ridentity = ffts[k]
        ft_ca = cfft(ca)
        ft_ra = rfft(ra)
        results[k]['sizes'].append(N)
        results[k]['complex_l2'].append(l2error(ft_ca, exact_ft_ca))
        results[k]['complex_l1'].append(l1error(ft_ca, exact_ft_ca))
        results[k]['complex_linf'].append(linferror(ft_ca, exact_ft_ca))
        results[k]['real_l2'].append(l2error(ft_ra, exact_ft_ra))
        results[k]['real_l1'].append(l1error(ft_ra, exact_ft_ra))
        results[k]['real_linf'].append(linferror(ft_ra, exact_ft_ra))
                
# calculate the identity error, i.e., error between x and ifft(fft(x))
# we can do this for larger array sizes where the calculation of the exact,
# high-precision DFT is infeasible.
for p in sorted(Path(config.DATA_DIR).iterdir()):
    if not p.is_file() or 'self' not in p.stem:
        continue
    with p.open('rb') as f:
        ca = np.load(f)
    ra = ca.real.copy() # simply take the real parts
    N = ca.shape[0]
    for k in ffts:
        cfft, cidentity, rfft, ridentity = ffts[k]
        ca2 = cidentity(ca) 
        ra2 = ridentity(ra)
        results[k]['identity_sizes'].append(N)
        results[k]['identity'].append(l2error(ca2, ca))
        results[k]['real_identity'].append(l2error(ra2, ra))

# save results
for k, v in ffts.items():
    res = results[k]
    with h5py.File(config.ACCURACY_RESULT_DIR + '/' + k + '.hdf5', 'w') as f:
        for k2 in res:
            res[k2] = np.array(res[k2])
        sidx = np.argsort(res['sizes'])
        sidx2 = np.argsort(res['identity_sizes'])
        for k2 in res:
            if 'identity' in k2:
                f.create_dataset(k2, data=res[k2][sidx2])
            else:
                f.create_dataset(k2, data=res[k2][sidx])
