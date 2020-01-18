# vtmm (vectorized transfer matrix method)

`vtmm` is a vectorized implementation of the [transfer matrix method](https://arxiv.org/abs/1603.02720) for computing the optical reflection and transmission of multilayer planar stacks. This package is in **beta**.

![](../master/img/spectrum_pcolor.png)

The `vtmm` package supports some of the same functionality as the [tmm](https://github.com/sbyrnes321/tmm) Python package developed by Steven Byrnes. However, in `vtmm` all operations are vectorized over angles / wavevectors as well as frequencies. Due to the small size of the matrices involved in the transfer matrix method (2 x 2), such vectorization results in significant performance gains, especially for large structures and many frequencies / wavevectors. In some cases we have observed an order of magnitude difference in execution time between the two implementations. The much lower execution time in `vtmm` may be useful for applications which require many evaluations of the reflection and transmission coefficients, such as in fitting or optimization.

## Gradients

Currently vtmm is implementing using a Tensor Flow backend. This means that gradients of scalar loss / objective functions of the transmission and reflection can be taken for "free." At a later time I will implement a numpy backend for users that do not need this functionality.

## Example

```
import tensorflow as tf
from vtmm import tmm_rt

pol = 's'
n = tf.constant([1.0, 3.5, 1.0])
d = tf.constant([2e-6])
kx = tf.linspace(0.0, 2*np.pi*220e12/299792458, 1000)
omega = tf.linspace(150e12, 220e12, 1000) * 2 * np.pi

t,r = tmm_rt(pol, omega, kx, n, d)
```
