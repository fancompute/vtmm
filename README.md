# vtmm (vectorized transfer matrix method)

`vtmm` is a vectorized implementation of the [transfer matrix method](https://arxiv.org/abs/1603.02720) for computing the optical reflection and transmission of multilayer planar stacks. This package is in **beta**.

![](../master/img/spectrum_pcolor.png)

The `vtmm` package supports some of the same functionality as the [tmm](https://github.com/sbyrnes321/tmm) Python package developed by Steven Byrnes. However, in `vtmm` all operations are vectorized over angles / wavevectors as well as frequencies. Due to the small size of the matrices involved in the transfer matrix method (2 x 2), such vectorization results in significant performance gains, especially for large structures and many frequencies / wavevectors. 

In some cases we have observed approximately two orders of magnitude difference in execution time between the two implementations (see below). The much lower execution time in `vtmm` may be useful for applications which require many evaluations of the reflection and transmission coefficients, such as in fitting or optimization.

## Gradients

Currently `vtmm` uses Tensor Flow as its backend. This means that gradients of scalar loss / objective functions of the transmission and reflection can be taken for free. At a later time a numpy backend may be implemented for users that do not need gradient functionality and/or do not want Tensor Flow as a requirement.

## Example

The entry point to `vtmm` is the function `tmm_rt(pol, omega, kx, n, d)`. See the example below for a basic illustration of how to use the package.

```python
import tensorflow as tf
from vtmm import tmm_rt

pol = 's'
n = tf.constant([1.0, 3.5, 1.0]) # Layer refractive indices 
d = tf.constant([2e-6]) # Layer thicknesses 
kx = tf.linspace(0.0, 2*np.pi*220e12/299792458, 1000) # Parallel wavevectors
omega = tf.linspace(150e12, 220e12, 1000) * 2 * np.pi # Angular frequencies

# t and r will be 2D tensors of shape [ num kx, num omega ]
t, r = tmm_rt(pol, omega, kx, n, d)
```

## Benchmarks

See `tests/test_benchmark.py` for a comparison between `vtmm` and the non-vectorized `tmm` package. The benchmarks shown below are for `len(omega) == len(kx) == 50` and 75 timeit evaluations.

```
python -W ignore ./tests/test_benchmark.py
```

```
Single omega / kx benchmark
vtmm: 0.2432 s
tmm:  0.0401 s

Large stack benchmark
vtmm: 0.7811 s
tmm:  79.8765 s

Medium stack benchmark
vtmm: 0.4607 s
tmm:  52.2255 s

Small stack benchmark
vtmm: 0.3367 s
tmm:  41.0926 s
```
