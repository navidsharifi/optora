"""Hypothesis strategies for ambiguity-set problem instances.

The strategies are biased towards the regimes that have broken an ambiguity
set before rather than towards generic random data:

- **ties** and **duplicate support points**: losses and support positions
  are drawn from small grids as often as from continuous ranges, so equal
  losses and coincident points (zero off-diagonal ground cost) are common;
- **zero weights**: nominal weights are drawn from a grid that contains
  zero, so scenarios with no nominal mass, including a zero-mass maximum,
  appear regularly;
- **`n = 1`**: the support size starts at one;
- **tiny radius**: radii are drawn log-uniformly down to `1e-14`, alongside
  exact zero and radii far past saturation.
"""

from typing import TypeVar

from hypothesis import strategies as st

from tests.dro.instances import Instance

MAX_SUPPORT_SIZE = 6

_GRID_WEIGHTS = (0.0, 0.0, 1.0, 1.0, 2.0, 3.0)
_GRID_LOSSES = (-1.0, 0.0, 1.0, 2.5)
_GRID_POSITIONS = (0.0, 1.0, 2.0, 3.0)
_SMALLEST_RADIUS_EXPONENT = -14.0
_LARGEST_RADIUS_EXPONENT = 3.0

_Element = TypeVar("_Element")

weights = st.one_of(
    st.sampled_from(_GRID_WEIGHTS), st.floats(min_value=1e-3, max_value=10.0)
)
losses = st.one_of(
    st.sampled_from(_GRID_LOSSES),
    st.floats(min_value=-10.0, max_value=10.0, allow_subnormal=False),
)
# Fine positions sit on a 1e-3 lattice: distinct points closer than that
# would only probe ground costs near the floating-point underflow threshold,
# not the geometry of the ambiguity set.
positions = st.one_of(
    st.sampled_from(_GRID_POSITIONS),
    st.integers(min_value=0, max_value=3000).map(lambda step: step / 1000.0),
)

radii = st.one_of(
    st.just(0.0),
    st.floats(
        min_value=_SMALLEST_RADIUS_EXPONENT, max_value=_LARGEST_RADIUS_EXPONENT
    ).map(lambda exponent: 10.0**exponent),
)
positive_radii = radii.filter(lambda radius: radius > 0.0)


def _vectors(
    elements: st.SearchStrategy[_Element], size: int
) -> st.SearchStrategy[list[_Element]]:
    """Return a strategy for lists of exactly `size` draws from `elements`."""
    return st.lists(elements, min_size=size, max_size=size)


@st.composite
def nominals(draw: st.DrawFn, support_size: int) -> tuple[float, ...]:
    """Draw a probability vector over `support_size` scenarios.

    Args:
        draw: Hypothesis draw function.
        support_size: Number of scenarios.

    Returns:
        Nonnegative weights normalized to sum to one, with at least one
        positive entry and frequent exact zeros.
    """
    raw = draw(_vectors(weights, support_size).filter(lambda values: sum(values) > 0.0))
    total = sum(raw)
    return tuple(value / total for value in raw)


def _loss_per_point(
    loss: list[float], position: tuple[float, ...]
) -> tuple[float, ...]:
    """Give every duplicate support point the loss of its first occurrence.

    A loss is a function of the support point, so coincident points must
    share it; two coincident points with different losses would let a
    Wasserstein ball move mass between them at zero cost, which is not a
    duplicate support point but a different geometry.

    Args:
        loss: Drawn per-scenario losses.
        position: Support positions, possibly with duplicates.

    Returns:
        The losses with each duplicate overwritten by its first occurrence.
    """
    first_loss: dict[float, float] = {}
    return tuple(
        first_loss.setdefault(point, value)
        for point, value in zip(position, loss, strict=True)
    )


@st.composite
def instances(
    draw: st.DrawFn, min_size: int = 1, max_size: int = MAX_SUPPORT_SIZE
) -> Instance:
    """Draw a nominal distribution, a loss vector, and support positions.

    Args:
        draw: Hypothesis draw function.
        min_size: Smallest support size to draw.
        max_size: Largest support size to draw.

    Returns:
        An `Instance` whose ground cost is the pairwise distance between
        its positions, so the cost has a zero diagonal and is zero between
        duplicate support points, which share their loss.
    """
    support_size = draw(st.integers(min_value=min_size, max_value=max_size))
    position = tuple(draw(_vectors(positions, support_size)))
    return Instance(
        name="generated",
        nominal=draw(nominals(support_size)),
        loss=_loss_per_point(draw(_vectors(losses, support_size)), position),
        position=position,
    )


@st.composite
def batches(
    draw: st.DrawFn, max_batch: int = 4
) -> tuple[tuple[Instance, ...], tuple[float, ...]]:
    """Draw a batch of instances sharing one support, each with its own radius.

    Args:
        draw: Hypothesis draw function.
        max_batch: Largest batch size to draw.

    Returns:
        Instances that share their support positions (and hence one ground
        cost) but have independent nominals and losses, and one radius per
        instance.
    """
    first = draw(instances())
    support_size = len(first.loss)
    batch_size = draw(st.integers(min_value=1, max_value=max_batch))
    members = (first,) + tuple(
        Instance(
            name="generated",
            nominal=draw(nominals(support_size)),
            loss=_loss_per_point(draw(_vectors(losses, support_size)), first.position),
            position=first.position,
        )
        for _ in range(batch_size - 1)
    )
    batch_radii = tuple(draw(_vectors(radii, batch_size)))
    return members, batch_radii
