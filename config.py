''' Global configuration file for the FFT benchmark.
'''
import os
import lib
from scipy.fftpack import next_fast_len

DATA_DIR = 'test_data'
ACCURACY_RESULT_DIR = 'accuracy_results'
SPEED_RESULT_DIR = 'speed_results'

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(ACCURACY_RESULT_DIR):
    os.mkdir(ACCURACY_RESULT_DIR)
if not os.path.exists(SPEED_RESULT_DIR):
    os.mkdir(SPEED_RESULT_DIR)

# The array sizes that are used in the accuracy test
# larger sizes may require more than 16GB of memory due to the high-precision
# calculations...

# 1st line : power-of-two
# 2nd line : Hamming (regular) numbers (those with prime factors <= 5)
# 3rd line : primes
ACCURACY_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
ACCURACY_SIZES += ([next_fast_len(n+1) for n in ACCURACY_SIZES] +
                   [lib.next_prime(n) for n in ACCURACY_SIZES])
ACCURACY_SIZES = sorted(ACCURACY_SIZES)
# The array sizes that are used in the identity accuracy test, i.e.,
# compare x and ifft(fft(x))).
ACCURACY_IDENTITY_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
# NumPy random number generator seed to create the same numbers
ACCURACY_SEED = 12345

# The array sizes that are used in the speed benchmark and also
SPEED_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
# add regular numbers and primes
SPEED_SIZES += ([next_fast_len(n+1) for n in SPEED_SIZES] +
                [lib.next_prime(n) for n in SPEED_SIZES])
SPEED_SIZES = sorted(SPEED_SIZES)
# NumPy random number generator seed to create the same numbers
SPEED_SEED = 12345
# The benchmark repeats the operation several times and takes the minimal time
SPEED_REPEAT = 20
