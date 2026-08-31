---
icon: lucide/play
---

# Examples

Reading a dual formula and believing it are two different things. Every
script in this section exists because, at some point while building Optora,
I wanted to *see* a formulation behave before trusting it: does the
worst-case expectation really saturate at $\max_i \mathrm{loss}_i$, does the
adversary really move mass the way the derivation claims, does the inner
dual solve really land where a brute-force grid search says it should?

So the four scripts under `examples/` are less "look how easy the API is"
and more small numerical experiments. Each one runs end to end, prints the
numbers it is checking, and draws a figure.

## Running them

The scripts plot, and `matplotlib` is deliberately not a runtime dependency
of `optora`, so it lives in its own extra:

```bash
pip install -e ".[examples]"
```

Then run whichever one you want from the repository root:

```bash
python examples/01_kl_dro_radius_sweep.py
```

!!! note "Where the figures go"

    Every figure is written to `examples/outputs/`, which is git-ignored.
    Nothing binary — no `.png`, no `.gif` — is ever committed to this
    repository; a plot that is worth publishing gets hosted externally and
    linked by URL. The reasoning is in
    [`examples/README.md`](https://github.com/navidsharifi/optora/blob/main/examples/README.md).

## The four experiments

| Example | Question it answers |
| --- | --- |
| [KL-DRO radius sweep](kl_dro_radius_sweep.md) | How does the worst-case expectation grow with the ambiguity radius, and does it respect both closed-form limits? |
| [Worst-case distribution shift](worst_case_distribution_shift.md) | Where does the adversary actually put the probability mass? |
| [Robust decisions across geometries](robust_decision_across_ambiguity_sets.md) | How much does the choice of ambiguity set change the decision you end up making? |
| [Convergence diagnostics](convergence_diagnostics.md) | Are the outer and the inner solve both genuinely converged, or only plausibly so? |

They are ordered roughly by how much machinery they involve: the first one
touches a single [`AmbiguitySet`](../api/core/dro_base.md), the last one
pulls a solver apart to inspect its trajectory.

!!! warning "These scripts are tuned down on purpose"

    Examples 3 and 4 nest an outer gradient descent around an ambiguity
    set's own inner dual solve, and in eager PyTorch that composition
    multiplies iteration counts unpleasantly fast. The budgets in those
    scripts are much smaller than each solver's defaults. They are chosen
    to show qualitative behaviour — monotonicity, limits, ordering — not to
    hit tight tolerances.

## The shared plotting helper

There is one non-example file in the folder. It only exists so that no
script has to think about where its figure goes:

```py title="examples/_plotting.py"
--8<-- "examples/_plotting.py"
```
