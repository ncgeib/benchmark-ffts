#!/bin/bash

python benchmark_accuracy.py numpy-1.15.0
python benchmark_accuracy.py numpy-pocketfft
python benchmark_accuracy.py scipy
python benchmark_accuracy.py pyfftw

python benchmark_speed.py numpy-1.15.0
python benchmark_speed.py numpy-pocketfft
python benchmark_speed.py scipy
python benchmark_speed.py pyfftw

python plot_accuracy_benchmark.py save
python plot_real_accuracy_benchmark.py save
python plot_speed_benchmark.py save
python plot_real_speed_benchmark.py save