r"""Maximum-mean-discrepancy-constrained ambiguity set (MMD DRO).

The worst case over a kernel ball,

$$
\sup \big\{ \ell^\top q \;:\; \mathbf{1}^\top q = 1,\; q \ge 0,\;
    \lVert q - p \rVert_K \le \epsilon \big\},
\qquad \lVert v \rVert_K = \sqrt{v^\top K v},
$$

is a second-order cone program with no one-dimensional structure: the
nonnegativity of $q$ couples every scenario through $K$, so no scalar
bracket pins the maximizer the way it does for the tilted sets. It is
solved by a primal-dual interior-point method on the cone
$\mathbb{R}^n_+ \times \mathcal{Q}^{n+1}$ with Nesterov-Todd scaling and
Mehrotra's predictor-corrector steps (Nesterov and Todd 1997; Mehrotra 1992;
Vandenberghe 2010), batched over the loss and radius, which converges in a
few tens of iterations however large the support.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from optora.core._transforms import transforms_active
from optora.core.convergence import ConvergenceTracker, validate_check_interval
from optora.core.dro_base import AmbiguitySet
from optora.divergences.mmd import MMDDivergence

# Fraction of the step to the cone boundary an iteration takes, so iterates
# stay strictly interior (Vandenberghe 2010).
_STEP_FRACTION = 0.99

# Iterative-refinement passes on the full linearized KKT system. The reduced
# normal equations lose accuracy as the iterates approach the boundary, and
# without refinement the equality residuals grow back once the duality gap
# is near the rounding floor.
_REFINEMENT_STEPS = 2

# Ratio of the residual tolerance to the duality-gap tolerance.
_RESIDUAL_SLACK = 10.0


def _soc_product(
    u0: torch.Tensor, u1: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the Jordan product $u \circ v = (u^\top v,\, u_0 v_1 + v_0 u_1)$."""
    return u0 * v0 + torch.sum(u1 * v1, dim=-1), u0.unsqueeze(-1) * v1 + (
        v0.unsqueeze(-1) * u1
    )


def _soc_divide(
    l0: torch.Tensor, l1: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Solve $\lambda \circ u = w$ for $u$, with $\lambda$ interior to the cone."""
    determinant = l0 * l0 - torch.sum(l1 * l1, dim=-1)
    u0 = (l0 * w0 - torch.sum(l1 * w1, dim=-1)) / determinant
    return u0, (w1 - u0.unsqueeze(-1) * l1) / l0.unsqueeze(-1)


def _soc_max_step(
    x0: torch.Tensor, x1: torch.Tensor, d0: torch.Tensor, d1: torch.Tensor
) -> torch.Tensor:
    r"""Return the largest $\alpha \ge 0$ with $x + \alpha d$ in the cone.

    With $\bar x = x / \sqrt{x^\top J x}$, the step is $1 / (\lVert \rho_1
    \rVert - \rho_0)$ for $\rho = P(\bar x^{-1/2})\, d / \sqrt{x^\top J x}$,
    the direction expressed relative to the cone's identity, and unbounded
    when that denominator is not positive (Vandenberghe 2010).
    """
    tiny = torch.finfo(x0.dtype).tiny
    root = torch.sqrt(torch.clamp(x0 * x0 - torch.sum(x1 * x1, dim=-1), min=tiny))
    b0 = x0 / root
    b1 = x1 / root.unsqueeze(-1)
    rho0 = (b0 * d0 - torch.sum(b1 * d1, dim=-1)) / root
    rho1 = (d1 - ((rho0 * root + d0) / (b0 + 1.0)).unsqueeze(-1) * b1) / root.unsqueeze(
        -1
    )
    inverse = torch.linalg.vector_norm(rho1, dim=-1) - rho0
    return torch.where(
        inverse > 0.0, torch.reciprocal(torch.clamp(inverse, min=tiny)), torch.inf
    )


def _orthant_max_step(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    r"""Return the largest $\alpha \ge 0$ with $x + \alpha d \ge 0$."""
    falling = d < 0.0
    ratio = torch.where(falling, -x / torch.where(falling, d, -1.0), torch.inf)
    return torch.amin(ratio, dim=-1)


class _Point(NamedTuple):
    r"""Primal and dual iterates, or a search direction of the same layout.

    `x` is the candidate distribution, `y` the multiplier of its total mass,
    `s1` and `z1` the orthant slack and its dual, and `(s20, s21)` and
    `(z20, z21)` the second-order-cone slack $(\epsilon, L^\top(x - p))$
    and its dual, split into scalar and vector parts.
    """

    x: torch.Tensor
    y: torch.Tensor
    s1: torch.Tensor
    s20: torch.Tensor
    s21: torch.Tensor
    z1: torch.Tensor
    z20: torch.Tensor
    z21: torch.Tensor

    def step(self, direction: "_Point", alpha: torch.Tensor) -> "_Point":
        """Return `self + alpha * direction`, per batch element."""
        vector = alpha.unsqueeze(-1)
        return _Point(
            *(
                value + (vector if value.ndim > alpha.ndim else alpha) * delta
                for value, delta in zip(self, direction, strict=True)
            )
        )


@dataclass(frozen=True)
class _Scaling:
    r"""Nesterov-Todd scaling of the cone at a primal-dual pair.

    On the orthant $W = \mathrm{diag}(\sqrt{s/z})$; on the second-order cone
    $W = \beta (2 v v^\top - J)$ with $J = \mathrm{diag}(1, -I)$, so that
    $W z = W^{-1} s = \lambda$, the scaled point (Vandenberghe 2010, sec. 4).
    """

    orthant: torch.Tensor
    v0: torch.Tensor
    v1: torch.Tensor
    beta: torch.Tensor

    @classmethod
    def at(cls, point: _Point) -> "_Scaling":
        """Compute the scaling at an interior primal-dual point."""
        tiny = torch.finfo(point.x.dtype).tiny
        s_norm = torch.sqrt(
            torch.clamp(point.s20**2 - torch.sum(point.s21**2, dim=-1), min=tiny)
        )
        z_norm = torch.sqrt(
            torch.clamp(point.z20**2 - torch.sum(point.z21**2, dim=-1), min=tiny)
        )
        s0, s1 = point.s20 / s_norm, point.s21 / s_norm.unsqueeze(-1)
        z0, z1 = point.z20 / z_norm, point.z21 / z_norm.unsqueeze(-1)
        gamma = torch.sqrt(0.5 * (1.0 + s0 * z0 + torch.sum(s1 * z1, dim=-1)))
        w0 = (s0 + z0) / (2.0 * gamma)
        w1 = (s1 - z1) / (2.0 * gamma).unsqueeze(-1)
        normalizer = torch.sqrt(2.0 * (w0 + 1.0))
        return cls(
            orthant=torch.sqrt(point.s1 / point.z1),
            v0=(w0 + 1.0) / normalizer,
            v1=w1 / normalizer.unsqueeze(-1),
            beta=torch.sqrt(s_norm / z_norm),
        )

    def soc(self, a0: torch.Tensor, a1: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the symmetric cone scaling $W$ to $(a_0, a_1)$."""
        inner = self.v0 * a0 + torch.sum(self.v1 * a1, dim=-1)
        return self.beta * (2.0 * self.v0 * inner - a0), self.beta.unsqueeze(-1) * (
            2.0 * self.v1 * inner.unsqueeze(-1) + a1
        )

    def soc_inverse(
        self, a0: torch.Tensor, a1: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        r"""Apply $W^{-1} = \beta^{-1} (2 J v v^\top J - J)$ to $(a_0, a_1)$."""
        inner = self.v0 * a0 - torch.sum(self.v1 * a1, dim=-1)
        return (2.0 * self.v0 * inner - a0) / self.beta, (
            -2.0 * self.v1 * inner.unsqueeze(-1) + a1
        ) / self.beta.unsqueeze(-1)


@dataclass(frozen=True)
class _Program:
    r"""The worst case as the cone program $\min c^\top x$ s.t. $G x + s = h$.

    The orthant block is $s_1 = x$ ($G_1 = -I$, $h_1 = 0$) and the cone block
    $s_2 = (\epsilon, L^\top (x - p))$ ($G_2 = [0; -L^\top]$, $h_2 =
    (\epsilon; -L^\top p)$), with $K = L L^\top$ and the equality
    $\mathbf{1}^\top x = 1$; $c = -\ell$.
    """

    loss: torch.Tensor
    factor: torch.Tensor
    gram: torch.Tensor
    image: torch.Tensor
    radius: torch.Tensor

    def cone_part(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return $L^\top x$ for a batch of vectors."""
        return x @ self.factor

    def cone_adjoint(self, z21: torch.Tensor) -> torch.Tensor:
        r"""Return $L z_{21}$ for a batch of vectors."""
        return z21 @ self.factor.mT

    def residuals(self, point: _Point) -> tuple[torch.Tensor, ...]:
        """Return the dual, equality, and cone residuals at `point`."""
        rx = (
            point.y.unsqueeze(-1)
            - point.z1
            - self.cone_adjoint(point.z21)
            - (self.loss)
        )
        ry = torch.sum(point.x, dim=-1) - 1.0
        rz1 = point.s1 - point.x
        rz20 = point.s20 - self.radius
        rz21 = point.s21 - self.cone_part(point.x) + self.image
        return rx, ry, rz1, rz20, rz21

    def initial_point(self) -> _Point:
        r"""Return the standard starting point (Vandenberghe 2010, sec. 5.3).

        It minimizes $\lVert G x - h \rVert^2 / 2 - \ell^\top x$ subject to
        the equality, then shifts the implied slack and dual into the cone.
        """
        normal = torch.eye(
            self.gram.shape[-1], dtype=self.gram.dtype, device=self.gram.device
        )
        cholesky = torch.linalg.cholesky(normal + self.gram)
        right = self.loss + self.cone_adjoint(self.image)
        particular = torch.cholesky_solve(right.unsqueeze(-1), cholesky).squeeze(-1)
        ones = torch.ones_like(particular)
        unit = torch.cholesky_solve(ones.unsqueeze(-1), cholesky).squeeze(-1)
        y = (torch.sum(particular, dim=-1) - 1.0) / torch.sum(unit, dim=-1)
        x = particular - y.unsqueeze(-1) * unit
        z1, z20, z21 = -x, -self.radius, -self.cone_part(x) + self.image
        s1, s20, s21 = -z1, -z20, -z21
        s1, s20 = _shift_into_cone(s1, s20, s21)
        z1, z20 = _shift_into_cone(z1, z20, z21)
        return _Point(x, y, s1, s20, s21, z1, z20, z21)


def _shift_into_cone(
    v1: torch.Tensor, v20: torch.Tensor, v21: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a multiple of the cone identity that makes `v` strictly interior."""
    excess = torch.maximum(
        -torch.amin(v1, dim=-1), torch.linalg.vector_norm(v21, dim=-1) - v20
    )
    shift = torch.where(excess >= 0.0, 1.0 + excess, torch.zeros_like(excess))
    return v1 + shift.unsqueeze(-1), v20 + shift


class _Solution(NamedTuple):
    """Maximizer and the multipliers of a solved worst case."""

    distribution: torch.Tensor
    orthant_dual: torch.Tensor
    radius_dual: torch.Tensor


def _solve(
    program: _Program, tol: float, max_iter: int, check_interval: int
) -> _Solution:
    """Run the predictor-corrector interior-point method on `program`.

    Args:
        program: Batched cone program, every tensor detached.
        tol: Tolerance on the relative duality gap; the residuals are held
            to `_RESIDUAL_SLACK` times it.
        max_iter: Maximum number of iterations.
        check_interval: Iterations between host reads of the convergence
            flag of the whole batch. Converged elements are frozen.

    Returns:
        The maximizer, projected onto the simplex to absorb the solve's
        residual, and the multipliers of the orthant and of the radius
        constraint.
    """
    point = program.initial_point()
    degree = program.loss.shape[-1] + 1
    data_norm = 1.0 + torch.sqrt(
        program.radius**2 + torch.sum(program.image**2, dim=-1)
    )
    done = torch.zeros(program.radius.shape, dtype=torch.bool, device=point.x.device)
    tracker = ConvergenceTracker(tol, check_interval, point.x)
    for iteration in range(max_iter):
        rx, ry, rz1, rz20, rz21 = program.residuals(point)
        gap = (
            torch.sum(point.s1 * point.z1, dim=-1)
            + point.s20 * point.z20
            + torch.sum(point.s21 * point.z21, dim=-1)
        )
        cost = -torch.sum(program.loss * point.x, dim=-1)
        primal = torch.sqrt(
            ry**2 + torch.sum(rz1**2, dim=-1) + rz20**2 + torch.sum(rz21**2, dim=-1)
        )
        dual = torch.linalg.vector_norm(rx, dim=-1)
        measure = torch.maximum(
            gap / (1.0 + cost.abs()),
            torch.maximum(primal / data_norm, dual / (1.0 + torch.amax(program.loss)))
            / _RESIDUAL_SLACK,
        )
        done = done | (measure <= tol)
        tracker.update(torch.amax(torch.where(done, 0.0, measure)))
        if tracker.should_stop(iteration):
            break
        direction = _search_direction(program, point, gap / degree)
        alpha = _STEP_FRACTION * _max_step(point, direction)
        finite = torch.isfinite(alpha)
        for delta in direction:
            entries = torch.isfinite(delta)
            finite = finite & (entries.all(-1) if delta.ndim > alpha.ndim else entries)
        alpha = torch.where(done | ~finite, 0.0, torch.clamp(alpha, max=1.0))
        point = point.step(_Point(*(torch.nan_to_num(d) for d in direction)), alpha)
    distribution = torch.clamp(point.x, min=0.0)
    return _Solution(
        distribution / torch.sum(distribution, dim=-1, keepdim=True),
        point.z1,
        point.z20,
    )


def _max_step(point: _Point, direction: _Point) -> torch.Tensor:
    """Return the largest step keeping the slack and the dual in the cone."""
    return torch.minimum(
        torch.minimum(
            _orthant_max_step(point.s1, direction.s1),
            _orthant_max_step(point.z1, direction.z1),
        ),
        torch.minimum(
            _soc_max_step(point.s20, point.s21, direction.s20, direction.s21),
            _soc_max_step(point.z20, point.z21, direction.z20, direction.z21),
        ),
    )


def _search_direction(program: _Program, point: _Point, mu: torch.Tensor) -> _Point:
    r"""Return Mehrotra's combined predictor-corrector direction.

    The affine-scaling predictor solves the Newton system for the
    complementarity target $\lambda \circ \lambda \to 0$; the corrector
    targets $\sigma \mu e$ with $\sigma = (1 - \alpha_{\mathrm{aff}})^3$ and
    adds the second-order term $(W^{-\top} \Delta s) \circ (W \Delta z)$
    of the predictor (Mehrotra 1992; Vandenberghe 2010, sec. 5.2).

    The Newton system reduces to the normal equations $H\, \Delta x +
    \mathbf{1}\, \Delta y = r$ with $H = G^\top W^{-2} G = \mathrm{diag}(z /
    s) + \beta^{-2} (K + 8 (1 + \lVert v_1 \rVert^2) (L v_1)(L v_1)^\top)$,
    one Cholesky factorization per iteration, and each solve is refined
    against the full system.
    """
    scaling = _Scaling.at(point)
    rx, ry, rz1, rz20, rz21 = program.residuals(point)
    lam1 = torch.sqrt(point.s1 * point.z1)
    lam20, lam21 = scaling.soc(point.z20, point.z21)
    cone_vector = program.cone_adjoint(scaling.v1)
    weight = 8.0 * (1.0 + torch.sum(scaling.v1**2, dim=-1)) / scaling.beta**2
    normal = program.gram / (scaling.beta**2)[..., None, None] + weight[
        ..., None, None
    ] * (cone_vector.unsqueeze(-1) * cone_vector.unsqueeze(-2))
    normal = normal + torch.diag_embed(point.z1 / point.s1)
    cholesky, _ = torch.linalg.cholesky_ex(normal)
    ones = torch.ones_like(point.x)
    unit = torch.cholesky_solve(ones.unsqueeze(-1), cholesky).squeeze(-1)

    def reduced(
        right_x: torch.Tensor,
        right_y: torch.Tensor,
        r1: torch.Tensor,
        r20: torch.Tensor,
        r21: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Solve G^T dz + 1 dy = right_x, 1^T dx = right_y, G dx - W^2 dz = r."""
        scaled1 = r1 / scaling.orthant**2
        scaled20, scaled21 = scaling.soc_inverse(*scaling.soc_inverse(r20, r21))
        right = right_x - scaled1 - program.cone_adjoint(scaled21)
        particular = torch.cholesky_solve(right.unsqueeze(-1), cholesky).squeeze(-1)
        dy = (torch.sum(particular, dim=-1) - right_y) / torch.sum(unit, dim=-1)
        dx = particular - dy.unsqueeze(-1) * unit
        dz1 = (-dx - r1) / scaling.orthant**2
        dz20, dz21 = scaling.soc_inverse(
            *scaling.soc_inverse(-r20, -program.cone_part(dx) - r21)
        )
        return dx, dy, dz1, dz20, dz21

    def newton(
        right_x: torch.Tensor,
        right_y: torch.Tensor,
        r1: torch.Tensor,
        r20: torch.Tensor,
        r21: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        solution = reduced(right_x, right_y, r1, r20, r21)
        for _ in range(_REFINEMENT_STEPS):
            dx, dy, dz1, dz20, dz21 = solution
            w20, w21 = scaling.soc(*scaling.soc(dz20, dz21))
            correction = reduced(
                right_x - (dy.unsqueeze(-1) * ones - dz1 - program.cone_adjoint(dz21)),
                right_y - torch.sum(dx, dim=-1),
                r1 - (-dx - scaling.orthant**2 * dz1),
                r20 - (-w20),
                r21 - (-program.cone_part(dx) - w21),
            )
            solution = tuple(
                value + delta for value, delta in zip(solution, correction, strict=True)
            )
        return solution

    def direction(
        target: torch.Tensor,
        correction1: torch.Tensor,
        correction20: torch.Tensor,
        correction21: torch.Tensor,
    ) -> tuple[_Point, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        square0, square1 = _soc_product(lam20, lam21, lam20, lam21)
        q1 = (target.unsqueeze(-1) - lam1 * lam1 - correction1) / lam1
        q20, q21 = _soc_divide(
            lam20, lam21, target - square0 - correction20, -square1 - correction21
        )
        t20, t21 = scaling.soc(q20, q21)
        dx, dy, dz1, dz20, dz21 = newton(
            -rx, -ry, -rz1 - scaling.orthant * q1, -rz20 - t20, -rz21 - t21
        )
        scaled_dz1 = scaling.orthant * dz1
        scaled_dz20, scaled_dz21 = scaling.soc(dz20, dz21)
        scaled_ds = (q1 - scaled_dz1, q20 - scaled_dz20, q21 - scaled_dz21)
        ds20, ds21 = scaling.soc(scaled_ds[1], scaled_ds[2])
        newton_step = _Point(
            dx, dy, scaling.orthant * scaled_ds[0], ds20, ds21, dz1, dz20, dz21
        )
        return newton_step, scaled_ds, (scaled_dz1, scaled_dz20, scaled_dz21)

    zero = torch.zeros_like(mu)
    affine, scaled_ds, scaled_dz = direction(
        zero, torch.zeros_like(lam1), zero, torch.zeros_like(lam21)
    )
    alpha = torch.clamp(_max_step(point, affine), max=1.0)
    sigma = (1.0 - alpha) ** 3
    second0, second1 = _soc_product(scaled_ds[1], scaled_ds[2], *scaled_dz[1:])
    combined, _, _ = direction(
        sigma * mu, scaled_ds[0] * scaled_dz[0], second0, second1
    )
    return combined


def _loss_hessian(
    gram: torch.Tensor,
    nominal: torch.Tensor,
    radius: torch.Tensor,
    solution: _Solution,
) -> torch.Tensor:
    r"""Return $\partial q^\star / \partial \ell$, the Hessian of the worst case.

    On the support $S$ of $q^\star$, identified by complementarity as the
    scenarios whose mass exceeds their orthant multiplier, the optimality
    conditions with the radius constraint active are

    $$
    \ell_S - \nu \mathbf{1} - \tilde\lambda\, [K (q - p)]_S = 0, \qquad
    \mathbf{1}^\top q = 1, \qquad (q - p)^\top K (q - p) = \epsilon^2,
    $$

    with $\tilde\lambda = \lambda / \epsilon$ and $q_i = 0$ off $S$. The
    implicit function theorem on this system gives the Jacobian of $q^\star$
    in $\ell$. Where the radius constraint is inactive (a saturated radius)
    the maximizer is locally constant and the Jacobian is zero.

    Args:
        gram: Gram matrix of shape `(n, n)`.
        nominal: Nominal of shape `(..., n)`.
        radius: Radius the program was solved at, of shape `batch_shape`.
        solution: The solved program.

    Returns:
        The symmetric positive semidefinite Jacobian, of shape
        `batch_shape + (n, n)`.
    """
    distribution = solution.distribution
    size = distribution.shape[-1]
    support = distribution > solution.orthant_dual
    rate = solution.radius_dual / radius
    deviation = (distribution - nominal) @ gram
    eye = torch.eye(size, dtype=gram.dtype, device=gram.device)
    rows = torch.where(support.unsqueeze(-1), -rate[..., None, None] * gram, eye)
    on_support = support.to(gram.dtype)
    column_nu = (-on_support).unsqueeze(-1)
    column_rate = (-on_support * deviation).unsqueeze(-1)
    top = torch.cat([rows, column_nu, column_rate], dim=-1)
    batch = distribution.shape[:-1]
    mass_row = torch.cat(
        [
            torch.ones(batch + (1, size), dtype=gram.dtype, device=gram.device),
            torch.zeros(batch + (1, 2), dtype=gram.dtype, device=gram.device),
        ],
        dim=-1,
    )
    norm_row = torch.cat(
        [
            2.0 * deviation.unsqueeze(-2),
            torch.zeros(batch + (1, 2), dtype=gram.dtype, device=gram.device),
        ],
        dim=-1,
    )
    jacobian = torch.cat([top, mass_row, norm_row], dim=-2)
    right = torch.cat(
        [
            -torch.diag_embed(on_support),
            torch.zeros(batch + (2, size), dtype=gram.dtype, device=gram.device),
        ],
        dim=-2,
    )
    # QR rather than LU: once `torch.set_num_threads` has been called, a
    # batched float32 LU on Windows MKL returns invalid pivots for these
    # systems, raising or silently giving nan, while QR factors each batch
    # element separately. A singular Jacobian yields a non-finite solution.
    orthogonal, triangular = torch.linalg.qr(jacobian)
    solved = torch.linalg.solve_triangular(
        triangular, orthogonal.mT @ right, upper=True
    )
    hessian = solved[..., :size, :]
    hessian = 0.5 * (hessian + hessian.mT)
    usable = (rate > 0.0) & torch.isfinite(hessian).all(-1).all(-1)
    return torch.where(usable[..., None, None], hessian, torch.zeros_like(hessian))


class MMDAmbiguitySet(AmbiguitySet):
    r"""Ambiguity set of distributions within a kernel MMD of the nominal (MMD DRO).

    For a nominal $p$ on a finite support $\{x_1, \dots, x_n\}$ and a
    positive semidefinite kernel with Gram matrix $K$, the set is the ball

    $$
    \mathcal{U} = \big\{ q \in \Delta_n \;:\;
        \mathrm{MMD}_k(q, p) = \lVert q - p \rVert_K \le \epsilon \big\}
    $$

    of the maximum mean discrepancy (Staib and Jegelka 2019; Zhu et al.
    2021), restricted to distributions on the support. Unlike a
    $\varphi$-divergence ball it contains distributions that move mass onto
    scenarios the nominal gives none, at a cost set by the kernel's
    geometry.

    The worst case $\sup_{q \in \mathcal{U}} \ell^\top q$ is solved exactly,
    as the second-order cone program it is, by a batched primal-dual
    interior-point method with Nesterov-Todd scaling (see the module
    docstring), to a relative duality gap of `tol` on the loss standardized
    to unit spread: in float64 the value is within about `tol` times the
    loss spread of the true worst case. In float32 the accuracy is set by
    the Gram matrix instead: factoring it in float32 costs up to about its
    condition number $\kappa(K)$ in units of float32 epsilon, relative to
    the loss spread, which for nearly coincident support points or a smooth
    kernel can exceed the robust excess itself. Pass `nominal`, `gram`, and
    `loss` in float64 when $\kappa(K)$ is large or the radius small; the
    gradient still reaches a float32 `loss` through the cast (see
    `docs/numerics.md`). Each iteration factors one
    `(n, n)` matrix per batch element, and a few tens of iterations
    suffice for supports into the thousands. The closed-form bound
    $\ell^\top p + \epsilon \lVert \ell \rVert_{K^{-1}}$ of Staib and
    Jegelka, which ignores $q \ge 0$, is not used: it overstates the worst
    case whenever the ball reaches the boundary of the simplex.

    The worst case is reported as $\sum_i q^\star_i \ell_i$ with $q^\star$
    detached, so its gradient in `loss` is the maximizer (Danskin's
    theorem). A zero-valued term adds its Hessian, the Jacobian
    $\partial q^\star / \partial \ell$ from the implicit function theorem on
    the optimality conditions over the maximizer's support, so second
    derivatives in `loss` are exact as well; it is built only when `loss`
    requires grad or the call runs inside a `torch.func` transform. The
    derivatives with respect to `nominal` and a tensor `radius` are first
    order, by the envelope theorem: the radius derivative is the multiplier
    of the radius constraint, which the solve resolves to about `tol`
    divided by the radius, relative to the loss spread.

    A radius at or beyond the kernel diameter of the simplex,
    $\max_{i,j} \lVert e_i - e_j \rVert_K$, admits every distribution, so
    the worst case is the largest loss; the solve caps the radius at twice
    that diameter plus one, which leaves the value unchanged and keeps the
    program well scaled.

    Attributes:
        nominal: Reference distribution the ambiguity set is centered on.
        divergence: `MMDDivergence` holding the Gram matrix; `contains` uses
            it.
        radius: Nonnegative bound $\epsilon$ on the maximum mean
            discrepancy, a float or a tensor of radii evaluated as one
            batch.
        kernel_factor: Factor $L$ with $L L^\top = K$, from an eigenvalue
            decomposition with negative rounding-level eigenvalues clipped,
            so a singular Gram matrix (duplicate support points) is valid.
        max_iter: Maximum number of interior-point iterations.
        tol: Tolerance on the relative duality gap, floored at 100 ulps of
            the working dtype.
        check_interval: Iterations between host reads of the convergence
            flag. An iteration factors a matrix per batch element, so a read
            every iteration is cheap by comparison.
    """

    divergence: MMDDivergence
    kernel_factor: torch.Tensor
    radius_ceiling: torch.Tensor

    def __init__(
        self,
        nominal: torch.Tensor,
        gram: torch.Tensor,
        radius: float | torch.Tensor,
        max_iter: int = 50,
        tol: float = 1e-10,
        check_interval: int = 1,
        validate: bool = False,
    ) -> None:
        r"""Initialize the MMD ambiguity set.

        Construction factors `gram` once by an eigenvalue decomposition,
        which may synchronize with the device.

        Args:
            nominal: Nominal distribution, a nonnegative tensor that sums to
                one along its last dimension.
            gram: Symmetric positive semidefinite Gram matrix of shape
                `(n, n)` of the kernel on the support points, for example
                the Laplacian kernel $e^{-\lVert x_i - x_j \rVert}$.
            radius: Nonnegative bound on the maximum mean discrepancy. A
                tensor radius is broadcast against the batch shape of
                `worst_case_expectation`'s `loss`.
            max_iter: Maximum number of interior-point iterations.
            tol: Tolerance on the relative duality gap.
            check_interval: Iterations between host reads of the
                convergence flag.
            validate: Whether to run the value checks that synchronize with
                the device on `nominal`, a tensor `radius`, and `gram`.

        Raises:
            ValueError: If `gram` does not match `nominal`'s support, a
                hyperparameter is invalid, `radius` is a negative float, or
                `validate` is set and an input is invalid.
        """
        if nominal.ndim > 0 and (gram.ndim != 2 or gram.shape[0] != nominal.shape[-1]):
            raise ValueError(
                f"gram must have shape ({nominal.shape[-1]}, {nominal.shape[-1]}) "
                f"to match nominal, got {tuple(gram.shape)}."
            )
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}.")
        if not tol > 0:
            raise ValueError(f"tol must be positive, got {tol}.")
        super().__init__(
            nominal=nominal,
            divergence=MMDDivergence(gram, validate=validate),
            radius=radius,
            validate=validate,
        )
        self.max_iter = max_iter
        self.tol = tol
        self.check_interval = validate_check_interval(check_interval)
        working = gram.to(torch.promote_types(gram.dtype, torch.float32))
        eigenvalues, vectors = torch.linalg.eigh(0.5 * (working + working.mT))
        self.register_buffer(
            "kernel_factor", vectors * torch.sqrt(torch.clamp(eigenvalues, min=0.0))
        )
        diagonal = torch.diagonal(working)
        squared = diagonal.unsqueeze(-1) + diagonal.unsqueeze(-2) - 2.0 * working
        diameter = torch.sqrt(torch.clamp(torch.amax(squared), min=0.0))
        self.register_buffer("radius_ceiling", 2.0 * diameter + 1.0)

    @property
    def gram(self) -> torch.Tensor:
        """Gram matrix of the kernel, aliasing `divergence.gram`."""
        return self.divergence.gram

    def worst_case_expectation(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute the worst-case expected loss over the MMD ball.

        Args:
            loss: Per-scenario loss values of shape `(..., n)`, one trailing
                entry per element of `nominal`'s support. Leading dimensions
                are a batch of independent loss vectors, solved together.

        Returns:
            A tensor of shape `(...)` holding the worst-case expected loss:
            the exact `sum(nominal * loss)` where the radius is zero, and
            otherwise the expectation of `loss` under the interior-point
            maximizer.

        Raises:
            ValueError: If `loss`'s trailing dimension does not match
                `nominal`'s support size, its batch shape does not broadcast
                against `nominal` and `radius`, or `validate` is set and
                `loss` is not finite.
        """
        batch_shape, working = self._prepare_loss(loss)
        if self._radius_is_zero:
            return self._as_result(
                self._nominal_expectation(working, batch_shape), loss
            )
        size = working.shape[-1]
        dtype = working.dtype
        detached = working.detach()
        low = torch.amin(detached, dim=-1, keepdim=True)
        spread = torch.amax(detached, dim=-1, keepdim=True) - low
        scale = torch.where(spread > 0.0, spread, torch.ones_like(spread))
        standardized = (working - low) / scale
        nominal = self._constant_nominal.to(dtype).expand(batch_shape + (size,))
        radius = torch.as_tensor(self.radius, dtype=dtype, device=working.device)
        radius = radius.detach().expand(batch_shape)
        ceiling = self.radius_ceiling.to(dtype)
        tol = max(self.tol, 100.0 * torch.finfo(dtype).eps)
        # A zero radius admits no strictly feasible point, so its element is
        # solved at a small probe radius instead, for its radius derivative.
        # The multiplier is accurate to about tol / radius and the
        # derivative's curvature bias grows with the radius, so the probe
        # sits at sqrt(tol) times the cap, between the two.
        solved_radius = torch.minimum(
            torch.where(radius > 0.0, radius, math.sqrt(tol) * ceiling), ceiling
        )
        gram = self.gram.to(dtype)
        factor = self.kernel_factor.to(dtype)
        with torch.no_grad():
            program = _Program(
                standardized.detach().expand(batch_shape + (size,)),
                factor,
                gram,
                nominal @ factor,
                solved_radius,
            )
            solution = _solve(program, tol, self.max_iter, self.check_interval)
        value = torch.sum(solution.distribution * standardized, dim=-1)
        if working.requires_grad or transforms_active():
            with torch.no_grad():
                hessian = _loss_hessian(gram, nominal, solved_radius, solution)
            shift = standardized - standardized.detach()
            value = value + 0.5 * torch.sum(
                shift * (hessian @ shift.unsqueeze(-1)).squeeze(-1), dim=-1
            )
        value = low.squeeze(-1) + scale.squeeze(-1) * value
        zero_radius = radius == 0.0
        value = torch.where(zero_radius, torch.sum(nominal * working, dim=-1), value)
        result = self._attach_parameter_gradients(
            value,
            lambda: self._parameter_sensitivities(
                detached,
                value.detach(),
                scale.squeeze(-1),
                nominal,
                gram,
                radius,
                solved_radius,
                solution,
            ),
        )
        return self._as_result(result, loss)

    def _parameter_sensitivities(
        self,
        loss: torch.Tensor,
        value: torch.Tensor,
        scale: torch.Tensor,
        nominal: torch.Tensor,
        gram: torch.Tensor,
        radius: torch.Tensor,
        solved_radius: torch.Tensor,
        solution: _Solution,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return the envelope derivatives with respect to `nominal` and `radius`.

        The radius derivative is the multiplier $\lambda^\star$ of the radius
        constraint, in loss units, and zero once the cap on the radius binds.
        The nominal enters only through the constraint, whose gradient in
        $p$ is $-K (q^\star - p) / \epsilon$, so the derivative is
        $\lambda^\star K (q^\star - p) / \epsilon$, shifted along the
        all-ones direction, where it is not defined, so that $\sum_i p_i\,
        \partial V / \partial p_i = V$. At a zero radius it is the loss
        itself, the derivative of $\langle p, \ell \rangle$, and the radius
        derivative is the right derivative, read off a solve at a small
        probe radius.

        Args:
            loss: Detached working-precision loss of shape `(..., n)`.
            value: Detached worst case of shape `batch_shape`.
            scale: Loss spread the program was standardized by.
            nominal: Detached nominal of shape `batch_shape + (n,)`.
            gram: Gram matrix in the working dtype.
            radius: Requested radius of shape `batch_shape`.
            solved_radius: Radius the program was solved at.
            solution: The solved program.

        Returns:
            The derivatives with respect to `nominal`, of shape
            `batch_shape + (n,)`, and `radius`, of shape `batch_shape`.
        """
        rate = solution.radius_dual / solved_radius
        deviation = (solution.distribution - nominal) @ gram
        gradient = (scale * rate).unsqueeze(-1) * deviation
        gradient = gradient + (value - torch.sum(nominal * gradient, dim=-1)).unsqueeze(
            -1
        )
        zero_radius = (radius == 0.0).unsqueeze(-1)
        gradient = torch.where(zero_radius, loss.expand_as(gradient), gradient)
        capped = radius >= self.radius_ceiling.to(radius.dtype)
        radius_gradient = torch.where(
            capped, torch.zeros_like(rate), scale * solution.radius_dual
        )
        return gradient, radius_gradient
