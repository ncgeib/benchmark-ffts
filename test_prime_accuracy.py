''' Tests the accuracy of the prime-size real-valued FFT.
'''
import numpy as np
import matplotlib.pyplot as plt
import lib


N = lib.next_prime(4096)
print(N)
#N = 256
np.random.seed(123)
ra = np.random.uniform(-0.5, 0.5, N)

ft_ra = lib.scipy_rfft(ra)
exact_ft_ra = lib.exact_rfft(ra)

plt.plot(np.abs(ft_ra - exact_ft_ra))
plt.show()
