''' Calculates the test arrays and the high-precision results of the FFTs.
'''

import numpy as np
from mpmath import mp
import lib
import config
import sys

# generate arrays and calculate their exact FFT using gmpy2
# save the array and its exact FFT
np.random.seed(config.ACCURACY_SEED)
for N in config.ACCURACY_SIZES:
    print('Calculating N=%d... ' % N, end='')
    sys.stdout.flush()
    ca = (np.random.uniform(-0.5, 0.5, N) +
          1.0j * np.random.uniform(-0.5, 0.5, N))
    ra = np.random.uniform(-0.5, 0.5, N)
    ft_ca = lib.exact_fft(ca)
    ft_ra = lib.exact_rfft(ra)
    with open(config.DATA_DIR + '/%d.npy' % N, 'wb') as f:
        np.save(f, ca)
        np.save(f, ft_ca)
        np.save(f, ra)
        np.save(f, ft_ra)
    print('Done.')


# generate larger arrrays and save them for the identity test
for N in config.ACCURACY_IDENTITY_SIZES:
    ca = (np.random.uniform(-0.5, 0.5, N) +
          1.0j * np.random.uniform(-0.5, 0.5, N))
    with open(config.DATA_DIR + '/self_%d.npy' % N, 'wb') as f:
        np.save(f, ca)
