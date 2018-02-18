## Synopsis

A collection of scripts to assess the accuracy and speed of different FFT implementations available in the Python packages NumPy, SciPy and pyfftw.

I only tested the code under Linux.

## Methodology

I tried to follow the benchmark methodology of the FFTW authors (http://www.fftw.org/benchfft/). The forward error of the FFT is calculated by comparing against a high-precision calculation of the DFT (implemented using gmpy2). Additionally, for larger array sizes the "identity error", i.e., the error between `x` and `ifft(fft(x))` is calculated. Note that a FFT implementation may have a large forward error while still reconstructing the original array very well.

## Usage

Most configuration options can be set in `config.py`. You can re-calculate the high-accuracy DFTs by by running
```
python prepare_accuracy_benchmark.py
```
The accuracy benchmark can be performed by running
```
python benchmark_accuracy.py numpy
```
where `numpy` can be a string containing `numpy`, `scipy` or `pyfftw` in which case the corresponding packages will be benchmarked and a file with the full label (e.g. `numpy-1.13.3`) will be created in `accuracy_results`. As a treat if you specify `fixed` in a label (e.g. `numpy-fixed`) it will load a numpy version from a path specified in the script (nice for testing against development versions).

The speed benchmark can be performed by running
```
python benchmark_speed.py numpy
```
where `numpy` can be replaced as described above.

The evaluation plots can be created by running
```
python plot[_real]_accuracy_benchmark.py [save]
python plot[_real]_speed_benchmark.py [save]
```

## Results
I looked at the results of NumPy 1.13.3, SciPy and pyfftw. Additionally I created a version of NumPy with correct constants and a more accurate calculation of the twiddle factors. One version calculated the twiddle factors in double precision (`numpy-fixed-double`) and one in extended precision (`numpy-fixed-long`).

[Complex FFT Accuracy](result_plots/accuracy_complex.png)
[Real FFT Accuracy](result_plots/accuracy_real.png)
[Complex FFT Speed](result_plots/speed_complex.png)
[Real FFT Speed](result_plots/speed_real.png)
