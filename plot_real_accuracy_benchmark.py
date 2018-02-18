''' Plots the results of the real FFT accuracy benchmark.
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
for p in Path(config.ACCURACY_RESULT_DIR).iterdir():
    if not p.is_file():
        continue
    results[p.stem] = h5py.File(str(p), mode='r')

# plot the results
fig, axs = plt.subplots(2, 2, figsize=(20.0/2.54, 20.0/2.54))
ax1, ax2, ax3, ax4 = axs.flat

print('Mean Error Real FFT (Non-Primes)')

i = 0
markers = ['>', '<', '+', 'x', '^']
styles = ['--', ':', '-.', '--', ':']
for k in sorted(results.keys()):
    v = results[k]
    kwargs = dict(label=k, marker=markers[i], ls=styles[i])
    n = np.array(v['sizes'][:], dtype=np.int)
    # the values which are a power of two
    f = (n != 0) & ((n & (n - 1)) == 0)
    # the primes
    fp = lib.is_prime(n)
    # the non-power of two, non-primes -> regular (or Hamming) numbers
    f2 = ~(f | fp)
    lc1, = ax1.plot(n[f], v['real_l2'][f], **kwargs)
    lc2, = ax2.plot(n[f2], v['real_l2'][f2], **kwargs)
    lc3, = ax3.plot(n[fp], v['real_linf'][fp], **kwargs)
    lc4, = ax4.plot(v['identity_sizes'], v['real_identity'], **kwargs)
    i += 1

    # ignore prime sizes as the error grows strongly with N
    print(k.ljust(20), np.mean(v['real_l2'][~fp]))

ax1.set_title('Power of Two')
ax2.set_title('Regular Numbers')
ax3.set_title('Primes')
ax4.set_title('Identity Error')

for ax in axs.flat:
    ax.set_yscale('log')
    ax.set_xscale('log', basex=2)
    ax.set_ylabel(r'$L_2$ error of $\mathcal{F}[x_n]$')
    ax.grid()
    ax.set_xlabel('array size')
ax4.set_ylabel(r'$L_2$ error of $\mathcal{F}^{-1}[\mathcal{F}[x_n]]$')
#ax1.set_ylim((5e-17, 5e-15))
ax1.legend(loc='best')
fig.suptitle('Real FFT')
fig.tight_layout()
fig.subplots_adjust(top=0.92)
if save:
    fig.savefig('result_plots/accuracy_real.png', dpi=300)
else:
    plt.show()
