# Convergence diagnostics

A DRO minimax problem has two solves stacked on top of each other, and
"it converged" can quietly mean two very different things. This example
takes them apart and checks each one separately, because a converged outer
loop sitting on top of an unconverged inner loop will still report a
perfectly confident-looking number.

## Outer: unroll the loop yourself

The first half stops calling
[`MinimaxSolver`](../api/dro/minimax_solver.md) as a black box and writes
out the same fixed-step iteration by hand:

$$
x_{k+1} = x_k - \alpha \, \nabla_x
\sup_{q \,:\, D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \le \rho}
\mathbb{E}_q\!\left[(\mathrm{outcomes} - x_k)^2\right]
$$

recording $x_k$ and the worst-case objective at every step. That gives the
trajectory the solver would otherwise throw away, which is what you actually
want when a run looks suspicious: a decision variable that has clearly
settled versus one still drifting is obvious in a plot and invisible in a
final scalar.

Note that each `torch.autograd.grad` call here differentiates *through* a
complete inner dual solve. It works, and it is cheap enough at this scale,
but it is worth appreciating that a single outer step is not a single unit of
work.

## Inner: brute-force the dual

The second half is the part I trust most. Optora reduces KL-DRO to a
one-dimensional convex dual,

$$
\inf_{\eta > 0} \; \eta \rho
+ \eta \log \mathbb{E}_{\mathrm{nominal}}\!\left[e^{\mathrm{loss}/\eta}\right],
$$

solved by gradient descent over $\log \eta$ so that $\eta > 0$ holds by
construction. Instead of believing that solve, the script evaluates the dual
objective on 400 points of a $\log \eta$ grid spanning
$[e^{-6}, e^{4}]$ and compares the grid minimum against what the library
returns.

Two independent methods, one of which has no tuning parameters at all. If
they agree, the dual solve is genuinely finding the adversarial
distribution; if they do not, the printed absolute difference tells you by
how much.

The grid evaluation is fully vectorized — one broadcast `logsumexp` over the
whole $(\text{grid} \times \text{scenarios})$ tensor, not a Python loop over
$\eta$ values.

!!! tip "Watch the shapes when you write your own grid search"

    The obvious way to broadcast a grid against a loss vector is also an
    easy way to accidentally build a $(G \times G)$ tensor and exhaust
    memory at a few million grid points. Check the shape of the intermediate
    before you widen the grid.

## Reading the figures

Two are produced. The first pair of axes shows the outer trajectory: $x$
against iteration, and the worst-case objective against iteration. The
second shows the inner dual landscape at the final decision — a clean convex
curve in $\log \eta$ with the grid-search minimizer marked. Seeing that
curve be smooth and single-troughed is itself reassuring; a kinked or flat
landscape would explain a lot of otherwise mysterious solver behaviour.

## Source

```py title="examples/04_convergence_diagnostics.py"
--8<-- "examples/04_convergence_diagnostics.py"
```
