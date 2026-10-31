---
icon: lucide/binary
---

# Numerical behaviour

This page states what Optora guarantees for each floating-point dtype and
what happens with non-finite inputs. It covers every
[ambiguity set](api/dro/index.md) and every
[divergence](api/divergences/index.md). Each row below is backed by the
tests it names.

## Dtype support

Every routine computes in a *working dtype*: the type promotion of its
inputs, raised to at least float32
([`working_dtype`](api/core/precision.md)). The result is returned in the
plain promotion of the inputs, without that floor, which is what any
PyTorch operation on them would return. For an ambiguity set the inputs are
`loss` and `nominal`. For a divergence they are `p` and `q`.

| Input dtype | Computed in | Returned as | Accuracy | Tested by |
| --- | --- | --- | --- | --- |
| `float64` | `float64` | `float64` | Reference. Agrees with independent LP and closed-form references to about $10^{-12}$. | `tests/dro/test_oracles.py` |
| `float32` | `float32` | `float32` | Within $128\,\varepsilon \cdot \max(\lvert v \rvert,\ \max \ell - \min \ell)$ of a float64 solve of the same inputs, where $v$ is the worst case. Typically a few $\varepsilon$. | `tests/dro/test_ambiguity_set_device_dtype.py` |
| `bfloat16` | `float32` | `bfloat16` | Exactly the float32 result rounded once. | `tests/dro/test_ambiguity_set_device_dtype.py`, `tests/divergences/test_divergence_device_dtype.py` |
| `float16` | `float32` | `float16` | Exactly the float32 result rounded once. | as `bfloat16` |
| mixed | the promotion, at least `float32` | the promotion | As its working dtype. | as `bfloat16` |

Here $\varepsilon$ is the machine epsilon of the input dtype:
$2^{-23} \approx 1.2 \times 10^{-7}$ for float32.

??? question "Why half precision is computed in float32"

    A worst case is a long reduction, a bisection whose trip count is read
    off the dtype, and differences of nearly equal sums. An 8-bit
    (`bfloat16`) or 11-bit (`float16`) significand loses most of that. Run
    natively, half precision was off by up to $2 \times 10^4$ units in the
    last place (Wasserstein, `float16`) and broke the bound
    $\mathbb{E}_{\mathrm{nominal}}[\ell] \le v \le \max \ell$. Computing in
    float32 and rounding once leaves only the final rounding. This is the
    same policy `torch.autocast` applies to `logsumexp`, `softmax` and
    `cumsum`.

Further guarantees, each tested:

- **Gradients.** The gradient with respect to a half-precision `loss` has
  the dtype of `loss`. It is exactly the float32 gradient rounded once.
- **Autocast.** `torch.autocast` does not change a worst case. The working
  dtype already decides the precision.
- **Half-precision sets.** A set whose buffers were cast with
  `.to(torch.bfloat16)` solves in float32 from those rounded buffers. A
  float32 set given a half-precision `loss` returns float32.

The float32 row, measured when this site is built on a random support of
256 scenarios, every set at seven fractions of its saturation radius:

<figure class="figure" markdown="span">
  ![Float32 error of every ambiguity set against a float64 solve of the same inputs, in units of the float32 machine epsilon](assets/figures/float32-errors-light.svg#only-light){ loading=lazy }
  ![Float32 error of every ambiguity set against a float64 solve of the same inputs, in units of the float32 machine epsilon](assets/figures/float32-errors-dark.svg#only-dark){ loading=lazy }
  <figcaption>Cells are $\lvert v_{32} - v_{64} \rvert / \max(\lvert v_{64}
  \rvert, \max \ell - \min \ell)$ in float32 ulps, the quantity the table
  bounds by 128. The MMD interior-point solve is the one set that is not
  held to that bound (see below).</figcaption>
</figure>

!!! warning "Two documented exceptions"

    - **Mass far below $\varepsilon$.** When the worst case concentrates on
      scenarios with tiny nominal mass, for example $10^{-20}$ in float32,
      the tilt sits near the top of its bisection bracket. The tilt rate is
      then resolved only to a relative $\varepsilon / (1 - t)$. KL and
      $\chi^2$ errors grow to about $10^{-4}$ relative (tested at
      $2048\,\varepsilon$ in `tests/dro/test_tilt_robustness.py`).
    - **`PhiAmbiguitySet`.** Its accuracy is set by the injected
      `dual_solver`, in every dtype. Only its dtype handling is guaranteed.
    - **`MMDAmbiguitySet` in float32.** The $128\,\varepsilon$ bound is
      tested for the tilt-solved and closed-form sets and for Wasserstein.
      The interior-point solve stops at a duality gap floored at 100 ulps,
      and in float32 the heatmap above measures errors of several hundred
      ulps at radii below one percent of saturation. Solve MMD balls in
      float64 when the radius is small.

### Supported loss range

A loss must be finite, and its spread $\max \ell - \min \ell$ must be finite
in the working dtype. In float32 that means $\lvert \ell \rvert$ below about
$1.7 \times 10^{38}$. A half-precision loss always qualifies, since it is
solved in float32. `WassersteinAmbiguitySet` also needs the spread divided
by the smallest positive cost difference to stay finite.

## Non-finite inputs and validation

Metadata checks are free and always run. Checks that read tensor values
back to the host block until the device has produced them, so they run only
with `validate=True`, in the spirit of `torch.distributions`'
`validate_args`.

| Input | Default (`validate=False`) | `validate=True` |
| --- | --- | --- |
| `nan` or $\pm\infty$ entry in `loss` | That batch element is `nan` or infinite, never a finite number. Every other batch element is unchanged. | `ValueError` on every call, at one host synchronization per call. |
| Negative or `nan` float `radius` | `ValueError` | `ValueError` |
| Infinite float `radius` | Allowed. The worst case is the largest reachable loss. | Allowed. |
| Negative or `nan` tensor `radius` | Not checked. The result is meaningless. | `ValueError` at construction. |
| Invalid `nominal` | Not checked. | `ValueError` at construction. |

Tested by `tests/dro/test_non_finite_losses.py` and
`tests/core/test_radius_and_loss_validation.py`.

Turn `validate=True` on while developing a model. Turn it off for training
on an accelerator, where a host synchronization per step costs throughput.

## Stability of the KL tilt

`KLAmbiguitySet` finds its worst case $q_\beta \propto
\mathrm{nominal} \cdot e^{-\beta\,\mathrm{gap}}$ by bisecting the tilt rate
$\beta$. The gap $\mathrm{gap} = (\max \ell - \ell)/(\max \ell - \min \ell)$
lies in $[0, 1]$ and is zero on the highest loss. Three choices keep every
step finite and accurate:

1. **Shifted exponent.** The largest exponent is exactly zero, which is the
   log-sum-exp shift, so no weight overflows however large $\beta$ becomes.
2. **Log-domain normalization.** $\log Z = \operatorname{logsumexp}(\log
   \mathrm{nominal} - \beta\,\mathrm{gap})$. A scenario whose nominal mass
   and weight would underflow as a product still gets its correct tilted
   mass.
3. **Two forms of the divergence.** Each is accurate in its own regime:

    $$
    D_{\mathrm{KL}}(q_\beta \,\|\, \mathrm{nominal}) =
    \begin{cases}
    -\log\!\big(1 + \mathbb{E}_{\mathrm{nominal}}[e^{c} - 1]\big),
        & D \le 1, \\[4pt]
    -\beta\,\mathbb{E}_{q_\beta}[\mathrm{gap}] - \log Z,
        & D > 1,
    \end{cases}
    \qquad c = -\beta\big(\mathrm{gap} - \mathbb{E}_{q_\beta}[\mathrm{gap}]\big).
    $$

    The first form, read through `expm1` and `log1p`, resolves a vanishing
    radius without cancellation. The second takes over for large
    divergences. There the first would represent $e^{-D}$ as one plus a sum
    near $-1$, which rounds to exactly $-1$ once $D$ exceeds about 16 in
    float32 or 36 in float64. The second form also takes over when the
    largest exponent $\beta\,\mathbb{E}_{q_\beta}[\mathrm{gap}]$ would
    overflow `expm1`. That needs a highest-loss nominal mass below
    $e^{-44}$ in float32.

    Choosing the form costs nothing extra per bisection step: both read
    quantities the tilt has already computed.
