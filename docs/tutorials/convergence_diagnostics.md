# Convergence diagnostics

A DRO minimax problem nests one solve inside another, so a statement that
the method converged can refer to either of them. This tutorial separates
the two and checks each one on its own, because a converged outer loop on
top of an unconverged inner solve still reports a value that looks
correct.

## The outer loop, unrolled by hand

The first half does not call
[`MinimaxSolver`](../api/dro/minimax_solver.md) as a black box. It writes
out the same fixed-step iteration by hand,

$$
x_{k+1} = x_k - \alpha \, \nabla_x
\sup_{q \,:\, D_{\mathrm{KL}}(q \,\|\, \mathrm{nominal}) \le \rho}
\mathbb{E}_q\!\left[(\mathrm{outcomes} - x_k)^2\right],
$$

and records $x_k$ and the worst-case objective at every step. This keeps
the trajectory that the solver would otherwise discard, which is the most
useful information when a run looks suspicious. A plot shows at once
whether the decision variable has settled or is still drifting, and a
final scalar does not.

Each evaluation of the objective runs a complete inner worst-case solve.
The backward pass of `torch.autograd.grad` does not differentiate through
that solve, because the gradient is the maximizing distribution (Danskin's
theorem). The cost is small at this scale, but one outer step still
contains a full inner solve.

## The inner solve, checked by a dual grid search

The second half checks the inner solve. KL-DRO has a one-dimensional
convex dual,

$$
\inf_{\eta > 0} \; \eta \rho
+ \eta \log \mathbb{E}_{\mathrm{nominal}}\!\left[e^{\mathrm{loss}/\eta}\right],
$$

whose value equals the worst-case expectation by strong duality
(Ben-Tal et al., 2013)[^btal]. Optora does not evaluate this dual. It
works in the primal and bisects the exponential tilt
$q_\beta \propto \mathrm{nominal} \cdot e^{\beta\,\mathrm{loss}}$ until the
KL constraint is tight. The dual is therefore an independent check. The
script evaluates it at 400 points of a $\log \eta$ grid, which covers
$\eta \in [e^{-6}, e^{4}]$, and compares the grid minimum with the value
the library returns.

Neither method has a step size or a tolerance to tune. If the two agree,
the library has found the worst-case distribution, and if they disagree,
the printed absolute difference shows by how much. The remaining
difference, of order $10^{-5}$, comes from the grid. Its spacing is about
$0.025$ in $\log \eta$, and the minimum over the grid lies above the
infimum by an amount that is quadratic in the distance from the minimizer
to the nearest grid point.

The grid evaluation is fully vectorized. It is one broadcast `logsumexp`
over the whole $(\text{grid} \times \text{scenarios})$ tensor, without a
Python loop over the values of $\eta$.

!!! tip "Watch the shapes when you write your own grid search"

    A common way to broadcast a grid against a loss vector can also build
    a $(G \times G)$ tensor by mistake and exhaust memory at a few million
    grid points. Check the shape of the intermediate tensor before you
    widen the grid.

## Reading the figures

The script produces two figures. The first has two panels for the outer
trajectory, with $x$ against the iteration on the left and the worst-case
objective against the iteration on the right. The second shows the inner
dual landscape at the final decision. It is a smooth curve in
$\log \eta$ with a single minimum, and the grid minimizer is marked. As
$\eta \to 0$ the dual objective approaches $\max_i \mathrm{loss}_i$, which
is why the curve flattens at the left end of the grid.

## Outer step, inner landscape

The two static figures show the outer trajectory and the inner landscape
at the end of the run, and the animation shows the steps in between. The
left panel shows the robust objective in $x$ with the iterate moving down
it, the middle panel shows the trajectory $x_k$, and the right panel shows
the inner dual landscape at the current $x_k$. The landscape changes with
every outer step, because the loss vector changes. The horizontal line at
the library's value meets the grid minimum at every outer step, including
the intermediate ones.

<figure class="figure" markdown="span">
  ![The outer iterate on the robust objective, its trajectory, and the inner KL dual landscape at every step](../assets/figures/convergence-diagnostics-light.gif#only-light){ loading=lazy }
  ![The outer iterate on the robust objective, its trajectory, and the inner KL dual landscape at every step](../assets/figures/convergence-diagnostics-dark.gif#only-dark){ loading=lazy }
  <figcaption>The left panel shows the robust objective and the nominal
  loss in $x$, with the iterate and its trail. The middle panel shows the
  decision at each outer iteration. The right panel shows the dual
  objective in $\log\eta$ at the current iterate, with the grid minimizer
  dashed and the library's worst-case value drawn as a horizontal
  line.</figcaption>
</figure>

## The figures

The figures are rendered from this script when the site is built, so they always show the output of the current code.

<figure class="figure" markdown="span">
  ![Outer gradient-descent trajectory of the robust decision and of the worst-case objective](../assets/figures/04_outer_convergence-light.png#only-light){ loading=lazy }
  ![Outer gradient-descent trajectory of the robust decision and of the worst-case objective](../assets/figures/04_outer_convergence-dark.png#only-dark){ loading=lazy }
  <figcaption>The decision variable and the worst-case objective at each outer iteration. Both level off within the 150 iterations, and the script prints their first and last values.</figcaption>
</figure>

<figure class="figure" markdown="span">
  ![KL-DRO dual objective over log eta at the robust solution](../assets/figures/04_inner_dual_landscape-light.png#only-light){ loading=lazy }
  ![KL-DRO dual objective over log eta at the robust solution](../assets/figures/04_inner_dual_landscape-dark.png#only-dark){ loading=lazy }
  <figcaption>The inner dual objective on a fine grid in $\log\eta$ at the robust solution, with the grid minimizer dashed. The script prints the grid minimum next to the library's worst-case value.</figcaption>
</figure>

## Source

```py title="tutorials/04_convergence_diagnostics.py"
--8<-- "tutorials/04_convergence_diagnostics.py"
```

[^btal]:
    Aharon Ben-Tal, Dick den Hertog, Anja De Waegenaere, Bertrand Melenberg
    and Gijs Rennen, "Robust Solutions of Optimization Problems Affected by
    Uncertain Probabilities", *Management Science* 59(2), 341-357 (2013).
