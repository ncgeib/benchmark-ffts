''' Plots the results of the complex FFT accuracy benchmark.
'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import config
import lib
import sys

# if the save flag is passed, the plot is saved to file
save = False
if len(sys.argv) > 1 and sys.argv[1] == 'save':
    save = True

# load all results
results = {}
for p in Path(config.SPEED_RESULT_DIR).iterdir():
    if not p.is_file():
        continue
    results[p.stem] = h5py.File(str(p), mode='r')

# plot the results
fig, axs = plt.subplots(2, 2, figsize=(20.0/2.54, 20.0/2.54))
ax1, ax2, ax3, ax4 = axs.flat

i = 0
markers = ['>', '<', '+', 'x', '^']
styles = ['--', ':', '-.', '--', ':']
for k in sorted(results.keys()):
    v = results[k]
    kwargs = dict(label=k, marker=markers[i], ls=styles[i])
    n = np.array(v['sizes'][:], dtype=np.int)
    s = np.array(v['complex_init'][:]) - np.array(v['complex'][:])
    # the values which are a power of two
    f = (n != 0) & ((n & (n - 1)) == 0)
    # the primes
    fp = lib.is_prime(n)
    # the non-power of two, non-primes -> regular (or Hamming) numbers
    f2 = ~(f | fp)
    key = 'complex'
    lc1, = ax1.plot(n[f], v[key][f], **kwargs)
    lc2, = ax2.plot(n[f2], v[key][f2], **kwargs)
    lc3, = ax3.plot(n[fp], v[key][fp], **kwargs)
#    if k != 'pyfftw':
#        lc4, = ax4.plot(n[~fp], s[~fp], **kwargs)
    i += 1

ax1.set_title('Power of Two')
ax2.set_title('Regular Numbers')
ax3.set_title('Primes')
#ax4.set_title('Twiddle Creation Time')

for ax in axs.flat[:-1]:
    ax.set_yscale('log')
    ax.set_xscale('log', basex=2)
    ax.set_ylabel('runtime [s]')
    ax.grid()
    ax.legend(loc='best')
    ax.set_xlabel('array size')
#ax4.set_ylabel('first runtime - best runtime [s]')
#ax1.set_ylim((8e-17, 2e-15))
#ax4.legend(loc='best')
fig.suptitle('Runtime Complex FFT')
fig.tight_layout()
fig.subplots_adjust(top=0.92)
if save:
    fig.savefig('result_plots/speed_complex.png', dpi=300)
else:
    plt.show()
