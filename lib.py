''' Helper functions for calculating the benchmarks.
'''
import time
import numpy as np
import scipy.fftpack
try:
    import gmpy2 #  much faster than using mpmath
    context = gmpy2.get_context()
    context.precision = 512 #  bits - should be very safe, one could reduce that
except:
    pass

def exact_fft(x, debug=False):
    ''' Calculates the exact DFT using gmpy2

    Has ~N^2 efficiency but I tried to optimize the calculation a bit.
    '''
    N = x.shape[0]
    start = time.clock()
    arg = -2j * (gmpy2.const_pi() / N)
    exp1d = np.array([gmpy2.exp(arg * idx) for idx in range(N)])
    exp = np.diag(exp1d)
    for i in range(N):
        for j in range(i+1):
            exp[j, i] = exp[i, j] = exp1d[(j * i) % N]
    stop = time.clock()
    if debug:
        print('preparation %.2fms' % ((stop-start)*1e3))

    start = time.clock()
    x = np.array([gmpy2.mpc(xi) for xi in x])
    # fsum solution - I noticed no difference except that it ran slower
#    y = []
#    for i in range(N):
#        mul = x * exp[i, :]
#        y.append(gmpy2.fsum([m.real for m in mul]) + 1j *
#                 gmpy2.fsum([m.imag for m in mul]))
#    y = np.array(y, dtype=np.complex128)
    y = np.sum(exp * x, axis=-1)
    y = np.array(y, dtype=np.complex128)
    stop = time.clock()
    if debug:
        print('calculation %.2fms' % ((stop-start)*1e3))
    return y

def exact_rfft(x, debug=False):
    ''' Calculates the exact real DFT using gmpy2.
    
    Just the first half of the complex output. 
    Too lazy to come up with something...
    '''
    N = x.shape[0]
    y = exact_fft(x, debug=debug)
    return y[:N//2+1]


def scipy_rfft(x):
    ''' Scipy returns the values in a different way than the other real FFTs,
    so this is a simple wrapper to facilitate direct comparisons.
    '''
    n = x.shape[0]
    fx = scipy.fftpack.rfft(x)
    y = np.zeros(x.shape[0]//2 + 1, dtype=np.complex128)
    y[0] = fx[0]
    if n % 2 == 0:
        y[-1] = fx[-1]
        y[1:-1] = fx[1:-1].view(np.complex128)
    else:
        y[1:] = fx[1:].view(np.complex128)
    return y


def is_prime_scalar(n):
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2))


is_prime = np.vectorize(is_prime_scalar)


def next_prime(n):
    while not is_prime_scalar(n):
        n += 1
    return n


def abs2(x):
    ''' More accurate (and faster) than np.abs(x)**2
    '''
    return x.real*x.real + x.imag*x.imag


def l2error(x, y):
    ''' Returns the normalized root mean square error.
    '''
    return np.sqrt(np.sum(abs2(y - x))/np.sum(abs2(y)))


def l1error(x, y):
    return np.abs(np.sum(np.abs(y - x))/np.sum(np.abs(y)))


def linferror(x, y):
    return np.max(np.abs(y - x))/np.max(np.abs(y))
