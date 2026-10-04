"""Training a model with KL-DRO under subpopulation shift.

Everything before this example evaluates DRO on a fixed loss vector. This one
trains a model: a linear regressor fitted by `torch.optim.Adam` on the
worst-case expected loss of a `KLAmbiguitySet`, against the same model fitted
by empirical risk minimization (ERM), and asks the question DRO is sold on.
When the test population mixes the training subpopulations in different
proportions, does the robust model degrade more gracefully?

Setup. Inputs are `x ~ N(0, I_2)` in both groups, so the group is latent and
cannot be recovered from `x`. The groups differ in the conditional law of the
label, `y = a_g . x + 0.3 * eps`:

* group A (majority, 90% of the training data) has slope `a_A = (1, 0)`;
* group B (minority, 10%) has slope `a_B = (1, 1.5)`.

Training uses
 = 400` points (360 from A, 40 from B), full batch, and the
model is
n.Linear(2, 1)` with a bias. The linear model is well specified for
the *training mixture*: nothing is misspecified. What changes at test time is
`P(y | x)`, through the mixing proportion `pi` of group B. This is the
textbook favourable case for DRO (subpopulation shift, Duchi and Namkoong
2021; Hashimoto et al. 2018), not a claim about DRO in general.

Because `x` has the same law in both groups, the risk of a model `(w, b)` on
group `g` is available exactly,

    R_g(w, b) = ||w - a_g||^2 + b^2 + sigma_g^2,

and the test risk at mixing proportion `pi` is the exact line
`R(pi) = (1 - pi) R_A + pi R_B`. No Monte Carlo sweep is needed; a large
shared test set is used only once, to cross-check the formula. The slope of
that line, `R_B - R_A`, is the model's *degradation rate*: how fast its risk
grows as the population moves from A toward B.

Models, all trained from the same data and initialization per seed, by the
same full-batch Adam loop (the loss is batched over seeds and models, so each
step is one joint solve):

* ERM: `KLAmbiguitySet(radius=0.0)`, the identical code path with no
  ambiguity;
* KL-DRO at radii 0.05, 0.2, 0.5;
* a ridge control: ERM plus `0.3 * ||w||^2`.

The radii and the ridge penalty were fixed before any shifted-test result was
looked at, and so were the slopes and noise level. A KL ball of radius `rho`
around the empirical distribution can move at most `m(rho)` of the probability
mass onto group B, with `m(0.05) ~ 0.21`, `m(0.2) ~ 0.33`, `m(0.5) ~ 0.50`
(computed below, not asserted by hand); the sample-level adversary never sees
the group labels, so it reaches only part of that.

Why the minority differs in a second feature rather than by an opposite sign
in one: with a single feature and opposite slopes, DRO is hard to tell from
shrinkage toward zero, which is exactly what weight decay does. Here the
population-optimal weights for mixing proportion `pi` are
`w*(pi) = (1 - pi) a_A + pi a_B`, a straight path that shrinkage cannot
follow, and the ridge control checks that it does not. For reference the
figure also draws a label-using ERM oracle that reweights group B explicitly;
sample-level DRO does not need labels, and the oracle only marks the frontier.

What to watch for:

* The exact crossing `pi*` per radius (DRO below ERM for `pi > pi*`), with
  its seed spread, and the cost to the left of it.
* The minority mass `q*` that the worst-case distribution actually uses,
  next to the mass the ball allows.
* A negative control with the identical pipeline: the minority differs only
  in noise (`sigma_B = 1`, same slope). DRO upweights those points as well,
  and gains nothing at any `pi`. It chases whatever is hard, not whatever is
  shifted.
* The ridge control: shrinkage moves the slope toward zero, which is away
  from both `a_A` and `a_B`, so at any penalty it is worse than ERM on both
  groups instead of trading one for the other. Measured against ERM, DRO
  moves mostly along the path `w*(pi)` and ridge mostly off it.

Training and convergence. The inner dual of each step is warm-started and
capped at 100 iterations, so it converges across steps while the Adam learning
rate decays; the gradient is then re-taken with a tightly solved dual
(`docs/training.md`: a loose inner `tol` biases the outer gradient). The
script asserts that ERM and ridge match their closed forms, that every
model's outer gradient is near zero, and that one DRO model matches an
independent joint L-BFGS solve over `(w, b, log eta)`.

Honest caveats. Every radius above zero costs in-distribution performance
(`pi = 0.1`), and DRO only wins once the test share of group B exceeds
`pi*`, which is above the training share. Risk can keep falling with the
radius at high `pi`, so the example does not claim an interior optimal
radius. Sample-level DRO is label-free but only partly upweights the
minority, and it is not the oracle: it reweights by loss, not by group, so
it need not match the oracle's trade-off at the same minority mass. With 10
seeds, individual per-seed numbers are noisy; the checks below use paired
per-seed differences, and the thresholds were calibrated from runs of this
script with slack.

References:
    John C. Duchi and Hongseok Namkoong, "Learning Models with Uniform
    Performance via Distributionally Robust Optimization", Annals of
    Statistics 49(3), 2021.
    Tatsunori Hashimoto, Megha Srivastava, Hongseok Namkoong and Percy Liang,
    "Fairness Without Demographics in Repeated Loss Minimization", ICML 2018.
    Shiori Sagawa, Pang Wei Koh, Tatsunori Hashimoto and Percy Liang,
    "Distributionally Robust Neural Networks for Group Shifts", ICLR 2020.

Run:
    python examples/07_subpopulation_shift_training.py
"""

import math
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from _plotting import save_figure
from torch import nn

from optora.dro import KLAmbiguitySet
from optora.solvers import GradientDescent

DTYPE = torch.float64
SEED = 20261004
NUM_SEEDS = 10
NUM_TRAIN = 400
MINORITY_FRACTION = 0.1
NUM_MINORITY = round(MINORITY_FRACTION * NUM_TRAIN)
RADII = (0.05, 0.2, 0.5)
RIDGE_PENALTY = 0.3
NOISE_STD = 0.3
NOISE_ONLY_MINORITY_STD = 1.0
SLOPE_MAJORITY = (1.0, 0.0)
SLOPE_MINORITY = (1.0, 1.5)
MIXING_GRID = torch.linspace(0.0, 1.0, 101, dtype=DTYPE)
ORACLE_GRID = torch.linspace(0.1, 0.9, 9, dtype=DTYPE)
NUM_TEST_PER_GROUP = 100_000
REPORTED_MIXING = (0.1, 0.5, 0.9)
REFERENCE_RADIUS = 0.2

# Thresholds of the seed-robust assertions, calibrated from runs of this
# script (see `check_shift_claims`) and set well inside the observed values.
HIGH_SHIFT = 0.9
HIGH_SHIFT_MARGIN = 0.25
IN_DISTRIBUTION_COST = 0.02
CROSSING_RANGE = (0.05, 0.6)
NOISE_ONLY_MEAN_COST = 0.002
NOISE_ONLY_SLACK = 1e-3

LEARNING_RATE = 0.2
LEARNING_RATE_DECAY = 0.98
NUM_STEPS = 300
INITIAL_LOG_ETA = 2.5
# The dual is not solved to convergence at every outer step: it is warm-started
# from the previous step, so it converges across steps while the learning rate
# decays. The final gradient is always taken with the precise solver below.
TRAINING_DUAL_SOLVER = GradientDescent(step_size=0.05, max_iter=100, tol=1e-9)
PRECISE_DUAL_SOLVER = GradientDescent(step_size=0.05, max_iter=20_000, tol=1e-12)

MODEL_NAMES = ("ERM", "ridge") + tuple(f"KL {radius}" for radius in RADII)
ERM, RIDGE = 0, 1
FIRST_DRO = 2


@dataclass(frozen=True)
class Scenario:
    """Population from which both groups are drawn.

    Attributes:
        slope_minority: Slope of the minority group's regression function.
        noise_minority: Label noise standard deviation of the minority group.
    """

    slope_minority: tuple[float, float]
    noise_minority: float

    def slopes(self) -> torch.Tensor:
        """Return the `(2, 2)` tensor of group slopes, majority first."""
        return torch.tensor([SLOPE_MAJORITY, self.slope_minority], dtype=DTYPE)

    def noise(self) -> torch.Tensor:
        """Return the `(2,)` tensor of group noise levels, majority first."""
        return torch.tensor([NOISE_STD, self.noise_minority], dtype=DTYPE)


SHIFT = Scenario(SLOPE_MINORITY, NOISE_STD)
NOISE_ONLY = Scenario(SLOPE_MAJORITY, NOISE_ONLY_MINORITY_STD)


@dataclass(frozen=True)
class Fit:
    """Trained models for one scenario, batched over seeds and models.

    Attributes:
        weight: Slopes of shape `(seeds, models, 2)`.
        bias: Intercepts of shape `(seeds, models)`.
        minority_mass: Probability the worst-case distribution puts on the
            minority points, shape `(seeds, radii)`.
        gradient_norm: Outer gradient norm at the returned parameters, shape
            `(seeds, models)`.
        x: Training inputs of shape `(seeds, n, 2)`.
        y: Training labels of shape `(seeds, n)`.
    """

    weight: torch.Tensor
    bias: torch.Tensor
    minority_mass: torch.Tensor
    gradient_norm: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor


class BatchedLinear(nn.Module):
    """Independent
    n.Linear(2, 1)` models stacked over seeds and models.

        Each seed's initialization is the default
    n.Linear` one, uniform on
        `[-1/sqrt(2), 1/sqrt(2)]`, shared by every model of that seed so that the
        only difference between models is the objective.

        Attributes:
            weight: Slopes of shape `(seeds, models, 2)`.
            bias: Intercepts of shape `(seeds, models)`.
    """

    def __init__(
        self, num_seeds: int, num_models: int, generator: torch.Generator
    ) -> None:
        """Initialize every model of a seed from one shared draw.

        Args:
            num_seeds: Number of independent training sets.
            num_models: Number of models trained per training set.
            generator: Seeded generator drawing the initialization.
        """
        super().__init__()
        bound = 1.0 / math.sqrt(2.0)
        start = (
            torch.rand(num_seeds, 3, generator=generator, dtype=DTYPE) * 2 - 1
        ) * bound
        self.weight = nn.Parameter(
            start[:, None, :2].expand(num_seeds, num_models, 2).clone()
        )
        self.bias = nn.Parameter(
            start[:, None, 2].expand(num_seeds, num_models).clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict with every model on every training set.

        Args:
            x: Inputs of shape `(seeds, n, 2)`.

        Returns:
            Predictions of shape `(seeds, models, n)`.
        """
        prediction: torch.Tensor = (
            torch.einsum("snd,smd->smn", x, self.weight) + self.bias[..., None]
        )
        return prediction


def sample_training_sets(
    scenario: Scenario, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one stratified training set per seed.

    The first `NUM_TRAIN - NUM_MINORITY` points belong to the majority and the
    rest to the minority; the model never sees this ordering.

    Args:
        scenario: Population the groups are drawn from.
        generator: Seeded generator.

    Returns:
        Inputs of shape `(seeds, n, 2)` and labels of shape `(seeds, n)`.
    """
    group = torch.zeros(NUM_TRAIN, dtype=torch.long)
    group[NUM_TRAIN - NUM_MINORITY :] = 1
    x = torch.randn(NUM_SEEDS, NUM_TRAIN, 2, generator=generator, dtype=DTYPE)
    noise = torch.randn(NUM_SEEDS, NUM_TRAIN, generator=generator, dtype=DTYPE)
    signal = torch.einsum("snd,nd->sn", x, scenario.slopes()[group])
    return x, signal + scenario.noise()[group] * noise


def minority_indicator() -> torch.Tensor:
    """Return the `(n,)` indicator of the minority training points."""
    indicator = torch.zeros(NUM_TRAIN, dtype=DTYPE)
    indicator[NUM_TRAIN - NUM_MINORITY :] = 1.0
    return indicator


def max_minority_mass(radius: float, fraction: float = MINORITY_FRACTION) -> float:
    """Return the most mass a KL ball can place on the minority group.

    Moving group mass from `fraction` to `m` while keeping the within-group
    proportions costs the binary divergence
    `m log(m / fraction) + (1 - m) log((1 - m) / (1 - fraction))`, which is
    the cheapest way to do it. The ball therefore allows group mass `m` if
    and only if this is at most `radius`. The sample-level adversary cannot
    exceed this and, not seeing the labels, stays below it.

    Args:
        radius: KL radius.
        fraction: Minority mass under the nominal distribution.

    Returns:
        The largest feasible minority mass, found by bisection.
    """
    low, high = fraction, 1.0 - 1e-12
    for _ in range(100):
        mid = 0.5 * (low + high)
        cost = mid * math.log(mid / fraction) + (1.0 - mid) * math.log(
            (1.0 - mid) / (1.0 - fraction)
        )
        low, high = (mid, high) if cost <= radius else (low, mid)
    return low


def least_squares(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    ridge: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve weighted, optionally ridge-penalized least squares in closed form.

    Args:
        x: Inputs of shape `(seeds, n, 2)`.
        y: Labels of shape `(seeds, n)`.
        weights: Sample weights of shape `(variants, n)`, each summing to one.
        ridge: Penalty on `||w||^2`; the intercept is not penalized.

    Returns:
        Slopes of shape `(seeds, variants, 2)` and intercepts of shape
        `(seeds, variants)`.
    """
    design = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    gram = torch.einsum("vn,snd,sne->svde", weights, design, design)
    gram = gram + ridge * torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE))
    moment = torch.einsum("vn,snd,sn->svd", weights, design, y)
    solution = torch.linalg.solve(gram, moment.unsqueeze(-1)).squeeze(-1)
    return solution[..., :2], solution[..., 2]


def group_risks(
    weight: torch.Tensor, bias: torch.Tensor, scenario: Scenario
) -> torch.Tensor:
    """Return the exact risk of each model on each group.

    Args:
        weight: Slopes of shape `(..., 2)`.
        bias: Intercepts of shape `(...)`.
        scenario: Population defining the group slopes and noise.

    Returns:
        A `(..., 2)` tensor of risks `||w - a_g||^2 + b^2 + sigma_g^2`,
        majority first.
    """
    distance = torch.sum((weight[..., None, :] - scenario.slopes()) ** 2, dim=-1)
    return distance + bias[..., None] ** 2 + scenario.noise() ** 2


def training_objective(
    model: BatchedLinear,
    x: torch.Tensor,
    y: torch.Tensor,
    erm_set: KLAmbiguitySet,
    dro_set: KLAmbiguitySet,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the summed training objective of every model.

    Models are independent, so summing their objectives gives each its own
    gradient, and Adam being elementwise makes the joint loop identical to
    training them one by one.

    Args:
        model: Batched linear models.
        x: Inputs of shape `(seeds, n, 2)`.
        y: Labels of shape `(seeds, n)`.
        erm_set: Zero-radius set evaluating the nominal empirical risk.
        dro_set: Positive-radius set, one radius per DRO model.

    Returns:
        The scalar objective and the `(seeds, radii, n)` per-sample loss the
        DRO set reduces.
    """
    loss = (model(x) - y[:, None, :]) ** 2
    dro_loss = loss[:, FIRST_DRO:]
    objective = (
        erm_set.worst_case_expectation(loss[:, :FIRST_DRO]).sum()
        + dro_set.worst_case_expectation(dro_loss).sum()
        + RIDGE_PENALTY * model.weight[:, RIDGE].square().sum()
    )
    return objective, dro_loss


def train(scenario: Scenario, generator: torch.Generator) -> Fit:
    """Train ERM, ridge and every KL-DRO radius on freshly drawn data.

    Args:
        scenario: Population the training sets are drawn from.
        generator: Seeded generator for the data and the initialization.

    Returns:
        The trained models and their diagnostics.
    """
    x, y = sample_training_sets(scenario, generator)
    model = BatchedLinear(NUM_SEEDS, len(MODEL_NAMES), generator)
    nominal = torch.full((NUM_TRAIN,), 1.0 / NUM_TRAIN, dtype=DTYPE)
    erm_set = KLAmbiguitySet(nominal, radius=0.0)
    dro_set = KLAmbiguitySet(
        nominal,
        radius=torch.tensor(RADII, dtype=DTYPE),
        dual_solver=TRAINING_DUAL_SOLVER,
        initial_log_eta=INITIAL_LOG_ETA,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, LEARNING_RATE_DECAY)
    for _ in range(NUM_STEPS):
        optimizer.zero_grad()
        objective, _ = training_objective(model, x, y, erm_set, dro_set)
        objective.backward()
        optimizer.step()
        scheduler.step()

    dro_set.dual_solver = PRECISE_DUAL_SOLVER
    optimizer.zero_grad()
    objective, dro_loss = training_objective(model, x, y, erm_set, dro_set)
    dro_loss.retain_grad()
    objective.backward()
    assert dro_loss.grad is not None
    assert model.weight.grad is not None
    assert model.bias.grad is not None
    gradient_norm = torch.sqrt(
        model.weight.grad.square().sum(-1) + model.bias.grad.square()
    )
    return Fit(
        weight=model.weight.detach(),
        bias=model.bias.detach(),
        minority_mass=dro_loss.grad @ minority_indicator(),
        gradient_norm=gradient_norm,
        x=x,
        y=y,
    )


def uniform_weights() -> torch.Tensor:
    """Return the `(1, n)` empirical (nominal) sample weights."""
    return torch.full((1, NUM_TRAIN), 1.0 / NUM_TRAIN, dtype=DTYPE)


def group_weights(minority_mass: torch.Tensor) -> torch.Tensor:
    """Return sample weights that put `minority_mass` on the minority group.

    Args:
        minority_mass: Total weight of group B, shape `(variants,)`.

    Returns:
        Weights of shape `(variants, n)`, uniform within each group. This
        uses the group labels and is only ever used by the oracle.
    """
    indicator = minority_indicator()
    majority = (1.0 - indicator) / (NUM_TRAIN - NUM_MINORITY)
    minority = indicator / NUM_MINORITY
    return (1.0 - minority_mass)[:, None] * majority + minority_mass[:, None] * minority


def oracle_risks(
    fit: Fit, scenario: Scenario, minority_mass: torch.Tensor
) -> torch.Tensor:
    """Return the exact group risks of group-reweighted ERM.

    Args:
        fit: Training sets to refit on.
        scenario: Population defining the exact risks.
        minority_mass: Minority weights to fit, shape `(variants,)`.

    Returns:
        Risks of shape `(seeds, variants, 2)`.
    """
    weight, bias = least_squares(fit.x, fit.y, group_weights(minority_mass))
    return group_risks(weight, bias, scenario)


def monte_carlo_group_risks(
    fit: Fit, scenario: Scenario, generator: torch.Generator
) -> torch.Tensor:
    """Estimate every model's group risks on a large shared test set.

    Args:
        fit: Trained models.
        scenario: Population the test set is drawn from.
        generator: Seeded generator.

    Returns:
        Estimated risks of shape `(seeds, models, 2)`, majority first.
    """
    num_seeds, num_models = fit.bias.shape
    x = torch.randn(NUM_TEST_PER_GROUP, 2, generator=generator, dtype=DTYPE)
    noise = torch.randn(NUM_TEST_PER_GROUP, 2, generator=generator, dtype=DTYPE)
    prediction = x @ fit.weight.reshape(-1, 2).T + fit.bias.reshape(-1)
    risks = []
    for group in range(2):
        label = x @ scenario.slopes()[group] + scenario.noise()[group] * noise[:, group]
        risks.append(((prediction - label[:, None]) ** 2).mean(0))
    return torch.stack(risks, dim=-1).reshape(num_seeds, num_models, 2)


def joint_reference(fit: Fit, radius: float) -> tuple[torch.Tensor, float]:
    """Solve the first seed's KL-DRO problem jointly over `(w, b, log eta)`.

    The dual objective is jointly convex in `(w, b, eta)` for a convex loss,
    so L-BFGS on all of it at once is an independent check on the nested
    Adam-plus-inner-solve training, from a different starting point.

    Args:
        fit: Trained models; only its first training set is used.
        radius: KL radius.

    Returns:
        The optimal `(w1, w2, b)` and the optimal worst-case risk.
    """
    x, y = fit.x[0], fit.y[0]
    weight, bias = least_squares(fit.x, fit.y, uniform_weights())
    start = torch.cat([weight[0, 0], bias[0], torch.tensor([1.0], dtype=DTYPE)])
    theta = start.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [theta],
        lr=1.0,
        max_iter=500,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def dual() -> torch.Tensor:
        eta = torch.exp(theta[3])
        loss = (x @ theta[:2] + theta[2] - y) ** 2
        return eta * radius + eta * (
            torch.logsumexp(loss / eta, dim=0) - math.log(NUM_TRAIN)
        )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        value = dual()
        value.backward()
        return value

    for _ in range(5):
        optimizer.step(closure)
    return theta.detach()[:3], float(dual().detach())


def path_geometry(
    weight: torch.Tensor, bias: torch.Tensor, scenario: Scenario
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate models relative to the population-optimal path `w*(pi)`.

    Args:
        weight: Slopes of shape `(..., 2)`.
        bias: Intercepts of shape `(...)`.
        scenario: Population defining the path.

    Returns:
        The mixing proportion of the closest path point and the distance to
        it, where the distance includes the intercept (zero on the path).
    """
    majority, minority = scenario.slopes()
    direction = minority - majority
    offset = weight - majority
    proportion = (offset @ direction / (direction @ direction)).clamp(0.0, 1.0)
    perpendicular = offset - proportion[..., None] * direction
    distance = torch.sqrt(perpendicular.square().sum(-1) + bias**2)
    return proportion, distance


def test_risk(risks: torch.Tensor, mixing: torch.Tensor) -> torch.Tensor:
    """Return the exact test risk at each mixing proportion.

    Args:
        risks: Group risks of shape `(..., 2)`.
        mixing: Proportions of group B, shape `(P,)`.

    Returns:
        A `(..., P)` tensor of `(1 - pi) R_A + pi R_B`.
    """
    return (1.0 - mixing) * risks[..., :1] + mixing * risks[..., 1:]


def crossing_proportion(risks: torch.Tensor) -> torch.Tensor:
    """Return where each DRO test-risk line crosses ERM's.

    With `cost = R_A(DRO) - R_A(ERM)` and `gain = R_B(ERM) - R_B(DRO)`, DRO is
    better than ERM exactly when `pi > cost / (cost + gain)`.

    Args:
        risks: Group risks of shape `(seeds, models, 2)`.

    Returns:
        Crossing proportions of shape `(seeds, radii)`.
    """
    difference = risks[:, FIRST_DRO:] - risks[:, ERM : ERM + 1]
    cost, gain = difference[..., 0], -difference[..., 1]
    return cost / (cost + gain)


def spread(values: torch.Tensor) -> str:
    """Format the mean and standard deviation over the seed dimension."""
    return f"{float(values.mean()):.3f} +/- {float(values.std()):.3f}"


def paired_gain(risks: torch.Tensor, mixing: float) -> torch.Tensor:
    """Return ERM's test risk minus every model's, seed by seed.

    Args:
        risks: Group risks of shape `(seeds, models, 2)`.
        mixing: Proportion of group B at test time.

    Returns:
        A `(seeds, models)` tensor; positive means the model beats ERM.
    """
    line = test_risk(risks, torch.tensor([mixing], dtype=DTYPE))[..., 0]
    return line[:, :1] - line


def displacement_from_erm(
    weight: torch.Tensor, bias: torch.Tensor, scenario: Scenario
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split each model's move away from ERM into along- and off-path parts.

    The path direction is `a_B - a_A`. Moving along it is what reweighting
    toward group B does; moving off it (in `w` orthogonally, or in the
    intercept) is what shrinkage or a bias does.

    Args:
        weight: Slopes of shape `(seeds, models, 2)`; model 0 is ERM.
        bias: Intercepts of shape `(seeds, models)`.
        scenario: Population defining the path direction.

    Returns:
        The signed along-path displacement and the unsigned off-path
        displacement, each of shape `(seeds, models)`.
    """
    majority, minority = scenario.slopes()
    direction = (minority - majority) / torch.linalg.vector_norm(minority - majority)
    move = weight - weight[:, ERM : ERM + 1]
    along = move @ direction
    off = torch.sqrt(
        (move.square().sum(-1) - along**2).clamp(min=0.0)
        + (bias - bias[:, ERM : ERM + 1]) ** 2
    )
    return along, off


def plot_results(
    risks: torch.Tensor,
    crossing: torch.Tensor,
    oracle_curve: torch.Tensor,
) -> None:
    """Draw the test-risk lines and the group-risk trade-off.

    Args:
        risks: Group risks of shape `(seeds, models, 2)`.
        crossing: Crossing proportions of shape `(seeds, radii)`.
        oracle_curve: Label-using reference risks of shape
            `(seeds, variants, 2)`, one per proportion of `ORACLE_GRID`.
    """
    fig, (ax_risk, ax_front) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["black", "tab:gray", "tab:blue", "tab:orange", "tab:red"]
    lines = test_risk(risks, MIXING_GRID)
    mean_line, std_line = lines.mean(0), lines.std(0)
    for index, name in enumerate(MODEL_NAMES):
        style = "--" if index == RIDGE else "-"
        ax_risk.plot(
            MIXING_GRID, mean_line[index], style, color=colors[index], label=name
        )
        ax_risk.fill_between(
            MIXING_GRID,
            mean_line[index] - std_line[index],
            mean_line[index] + std_line[index],
            color=colors[index],
            alpha=0.12,
        )
    ax_risk.axvline(MINORITY_FRACTION, color="black", linestyle=":", linewidth=1)
    ax_risk.annotate(
        "training mix",
        (MINORITY_FRACTION, 0.0),
        xytext=(4, 6),
        textcoords="offset points",
        fontsize="small",
    )
    erm_mean = risks[:, ERM].mean(0)
    for index in range(len(RADII)):
        position = float(crossing[:, index].mean())
        height = float((1 - position) * erm_mean[0] + position * erm_mean[1])
        ax_risk.errorbar(
            position,
            height,
            xerr=float(crossing[:, index].std()),
            fmt="o",
            color=colors[FIRST_DRO + index],
            markeredgecolor="black",
            capsize=3,
            zorder=5,
        )
    ax_risk.set_xlabel("proportion of group B at test time, $\\pi$")
    ax_risk.set_ylabel("exact test risk")
    ax_risk.set_title("DRO crosses ERM at $\\pi^*$ (dots) and wins beyond it")
    ax_risk.legend(fontsize="small")

    mixing = torch.linspace(0.0, 1.0, 101, dtype=DTYPE)
    gap = float(torch.sum((SHIFT.slopes()[1] - SHIFT.slopes()[0]) ** 2))
    noise_a, noise_b = (float(value) for value in SHIFT.noise() ** 2)
    ax_front.plot(
        mixing**2 * gap + noise_a,
        (1 - mixing) ** 2 * gap + noise_b,
        color="tab:green",
        label="population-optimal path $w^*(\\pi)$",
    )
    oracle_mean = oracle_curve.mean(0)
    ax_front.plot(
        oracle_mean[:, 0],
        oracle_mean[:, 1],
        "--",
        color="tab:purple",
        label="group-reweighted ERM (uses labels)",
    )
    for index, name in enumerate(MODEL_NAMES):
        ax_front.errorbar(
            float(risks[:, index, 0].mean()),
            float(risks[:, index, 1].mean()),
            xerr=float(risks[:, index, 0].std()),
            yerr=float(risks[:, index, 1].std()),
            fmt="o",
            color=colors[index],
            capsize=3,
            label=name,
        )
    ax_front.set_xlabel("majority risk $R_A$")
    ax_front.set_ylabel("minority risk $R_B$")
    ax_front.set_title("DRO moves along the optimal trade-off path")
    ax_front.legend(fontsize="small")
    fig.tight_layout()
    save_figure(fig, "07_subpopulation_shift_training")


def report_group_risks(
    fit: Fit,
    risks: torch.Tensor,
    oracle: torch.Tensor,
    bounds: list[float],
) -> None:
    """Print the per-model group risks and worst-case minority masses.

    Args:
        fit: Trained models.
        risks: Group risks of shape `(seeds, models, 2)`.
        oracle: Label-using reference risks of shape `(seeds, radii, 2)`.
        bounds: Largest minority mass each radius allows.
    """
    print("\nmean over seeds   R_A    R_B  R_B-R_A  worst  q*(B)  allowed")
    mass = [MINORITY_FRACTION] * FIRST_DRO + fit.minority_mass.mean(0).tolist()
    allowed = [None] * FIRST_DRO + bounds
    for index, name in enumerate(MODEL_NAMES):
        group_a, group_b = (float(value) for value in risks[:, index].mean(0))
        cap = "" if allowed[index] is None else f"{allowed[index]:.3f}"
        print(
            f"{name:<15} {group_a:>6.3f} {group_b:>6.3f} {group_b - group_a:>8.3f}"
            f" {max(group_a, group_b):>6.3f} {mass[index]:>6.3f} {cap:>8}"
        )
    for index, bound in enumerate(bounds):
        group_a, group_b = (float(value) for value in oracle[:, index].mean(0))
        print(
            f"oracle q={bound:.2f}   {group_a:>6.3f} {group_b:>6.3f}"
            f" {group_b - group_a:>8.3f} {max(group_a, group_b):>6.3f}"
            f" {bound:>6.2f}   <- uses group labels"
        )


def report_test_risk(risks: torch.Tensor) -> None:
    """Print test risk and the paired gain over ERM at selected proportions.

    Args:
        risks: Group risks of shape `(seeds, models, 2)`.
    """
    print("\ntest risk, mean over seeds (paired gain of ERM - model in brackets)")
    print("pi    " + "".join(f"{name:>19}" for name in MODEL_NAMES))
    for pi in REPORTED_MIXING:
        line = test_risk(risks, torch.tensor([pi], dtype=DTYPE))[..., 0]
        gain = paired_gain(risks, pi)
        cells = "".join(
            f"{float(line[:, index].mean()):>10.3f}"
            f" ({float(gain[:, index].mean()):+.3f})"
            for index in range(len(MODEL_NAMES))
        )
        print(f"{pi:<5} {cells}")
    crossing = crossing_proportion(risks)
    print("\nDRO beats ERM for pi > pi* (mean +/- std over seeds)")
    for index, radius in enumerate(RADII):
        print(f"  KL {radius:<5} pi* = {spread(crossing[:, index])}")


def report_geometry(fit: Fit, risks: torch.Tensor, bounds: list[float]) -> None:
    """Print where each model sits relative to the population-optimal path.

    Args:
        fit: Trained models.
        risks: Group risks of shape `(seeds, models, 2)`.
        bounds: Largest minority mass each radius allows.
    """
    proportion, distance = path_geometry(fit.weight, fit.bias, SHIFT)
    along, off = displacement_from_erm(fit.weight, fit.bias, SHIFT)
    print("\nmean over seeds   pi_proj  dist. to path  along-path move  off-path move")
    for index, name in enumerate(MODEL_NAMES):
        print(
            f"{name:<15} {float(proportion[:, index].mean()):>8.3f}"
            f" {float(distance[:, index].mean()):>14.3f}"
            f" {float(along[:, index].mean()):>16.3f}"
            f" {float(off[:, index].mean()):>14.3f}"
        )
    print(
        "sample-level minority mass q* of the DRO models:",
        [f"{value:.3f}" for value in fit.minority_mass.mean(0).tolist()],
        "vs allowed",
        [f"{bound:.3f}" for bound in bounds],
    )
    assert risks.shape[1] == len(MODEL_NAMES)


def check_training(fit: Fit, risks: torch.Tensor, generator: torch.Generator) -> None:
    """Check that the trained models are converged and the risks are exact.

    Under-converged training must not be able to fake the result, so the
    ERM and ridge solutions are compared with their closed forms, the outer
    gradient of every model (taken with a precisely solved dual) must be
    small, one DRO model is compared with an independent joint L-BFGS solve,
    and the exact risk formula is compared with a large shared test set.

    Args:
        fit: Trained models of the shifted scenario.
        risks: Their exact group risks.
        generator: Seeded generator for the Monte Carlo test set.
    """
    erm_weight, erm_bias = least_squares(fit.x, fit.y, uniform_weights())
    ridge_weight, ridge_bias = least_squares(
        fit.x, fit.y, uniform_weights(), ridge=RIDGE_PENALTY
    )
    erm_error = max(
        float((fit.weight[:, ERM] - erm_weight[:, 0]).abs().max()),
        float((fit.bias[:, ERM] - erm_bias[:, 0]).abs().max()),
    )
    ridge_error = max(
        float((fit.weight[:, RIDGE] - ridge_weight[:, 0]).abs().max()),
        float((fit.bias[:, RIDGE] - ridge_bias[:, 0]).abs().max()),
    )
    gradient = float(fit.gradient_norm.max())
    reference_index = FIRST_DRO + RADII.index(REFERENCE_RADIUS)
    reference, _ = joint_reference(fit, REFERENCE_RADIUS)
    trained = torch.cat(
        [
            fit.weight[0, reference_index],
            fit.bias[0, reference_index : reference_index + 1],
        ]
    )
    reference_error = float((reference - trained).abs().max())
    monte_carlo_error = float(
        (monte_carlo_group_risks(fit, SHIFT, generator) - risks).abs().max()
    )

    print("\nconvergence and exactness checks")
    print(f"  Adam ERM   vs closed form       max |diff| = {erm_error:.2e}")
    print(f"  Adam ridge vs closed form       max |diff| = {ridge_error:.2e}")
    print(f"  outer gradient norm, all models max        = {gradient:.2e}")
    print(
        f"  KL {REFERENCE_RADIUS} vs joint L-BFGS (seed 0)   max |diff| = "
        f"{reference_error:.2e}"
    )
    print(
        f"  exact risk vs {NUM_TEST_PER_GROUP} points    max |diff| = "
        f"{monte_carlo_error:.2e}"
    )
    assert erm_error < 1e-3 and ridge_error < 1e-3
    assert gradient < 1e-3
    assert reference_error < 1e-3
    assert monte_carlo_error < 0.02

    majority_risk, minority_risk = risks[..., 0], risks[..., 1]
    gap = float(torch.sum((SHIFT.slopes()[1] - SHIFT.slopes()[0]) ** 2))
    noise_a, noise_b = (float(value) for value in SHIFT.noise() ** 2)
    proportion = ((majority_risk - noise_a) / gap).clamp(max=1.0).sqrt()
    frontier = (1.0 - proportion) ** 2 * gap + noise_b
    assert bool((minority_risk >= frontier - 1e-9).all())


def check_shift_claims(fit: Fit, risks: torch.Tensor, bounds: list[float]) -> None:
    """Assert the paired, seed-robust claims about the shifted scenario.

    Every comparison is a per-seed difference between models trained on the
    same data, so seed noise in the data cancels. Thresholds were calibrated
    from runs of this script with slack; they are deliberately far from the
    observed values and are not claims about exact numbers. In particular
    nothing asserts an interior optimal radius.

    Args:
        fit: Trained models.
        risks: Their exact group risks.
        bounds: Largest minority mass each radius allows.
    """
    dro = slice(FIRST_DRO, None)
    shifted = paired_gain(risks, HIGH_SHIFT)[:, dro]
    assert bool((shifted.mean(0) > HIGH_SHIFT_MARGIN).all())
    assert bool((shifted > HIGH_SHIFT_MARGIN / 2).all())

    in_distribution = -paired_gain(risks, MINORITY_FRACTION)[:, dro]
    assert bool((in_distribution.mean(0) > IN_DISTRIBUTION_COST).all())
    assert bool((in_distribution > 0.0).all())

    rate = risks[..., 1] - risks[..., 0]
    ordered_rate = rate[:, [ERM, *range(FIRST_DRO, len(MODEL_NAMES))]]
    assert bool((torch.diff(ordered_rate, dim=1) < 0.0).all())

    mass = fit.minority_mass
    assert bool((mass > MINORITY_FRACTION).all())
    assert bool((torch.diff(mass, dim=1) > 0.0).all())
    assert bool((mass < torch.tensor(bounds, dtype=DTYPE)).all())

    crossing = crossing_proportion(risks).mean(0)
    assert bool(((crossing > CROSSING_RANGE[0]) & (crossing < CROSSING_RANGE[1])).all())

    # Shrinkage moves a regression away from a_B and from a_A, so the ridge
    # control is dominated by ERM on both groups rather than trading them.
    assert bool((risks[:, RIDGE] > risks[:, ERM]).all())

    along, off = displacement_from_erm(fit.weight, fit.bias, SHIFT)
    assert bool((along[:, dro].mean(0) > off[:, dro].mean(0)).all())
    assert bool(off[:, RIDGE].mean() > along[:, RIDGE].abs().mean())


def check_noise_control(control: Fit, control_risks: torch.Tensor) -> None:
    """Print and assert the noise-only negative control.

    The minority has the same regression function and only noisier labels,
    so there is no shift for DRO to protect against. Its adversary still
    upweights those points, and the resulting models are worse than ERM on
    both groups, hence at every mixing proportion.

    Args:
        control: Models trained on the noise-only scenario.
        control_risks: Their exact group risks.
    """
    difference = control_risks[:, FIRST_DRO:] - control_risks[:, ERM : ERM + 1]
    print("\nnegative control: minority differs only in noise (sigma_B = 1)")
    print("  radius   q*(B)   mean(R_DRO - R_ERM) on A, B   worst seed")
    for index, radius in enumerate(RADII):
        mean_a, mean_b = (float(value) for value in difference[:, index].mean(0))
        print(
            f"  {radius:<7} {float(control.minority_mass[:, index].mean()):.3f}"
            f"   {mean_a:>+10.4f} {mean_b:>+10.4f}"
            f"   {float(difference[:, index].min()):>+10.4f}"
        )
    assert bool((difference.mean(0) > NOISE_ONLY_MEAN_COST).all())
    assert bool((difference > -NOISE_ONLY_SLACK).all())
    assert bool((control.minority_mass > MINORITY_FRACTION).all())


def main() -> None:
    start_time = time.perf_counter()
    generator = torch.Generator().manual_seed(SEED)
    fit = train(SHIFT, generator)
    control = train(NOISE_ONLY, generator)

    risks = group_risks(fit.weight, fit.bias, SHIFT)
    control_risks = group_risks(control.weight, control.bias, NOISE_ONLY)
    bounds = [max_minority_mass(radius) for radius in RADII]
    oracle = oracle_risks(fit, SHIFT, torch.tensor(bounds, dtype=DTYPE))
    oracle_curve = oracle_risks(fit, SHIFT, ORACLE_GRID)

    check_training(fit, risks, generator)
    report_group_risks(fit, risks, oracle, bounds)
    report_test_risk(risks)
    report_geometry(fit, risks, bounds)
    check_shift_claims(fit, risks, bounds)
    check_noise_control(control, control_risks)
    plot_results(risks, crossing_proportion(risks), oracle_curve)
    print(f"\nfinished in {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
