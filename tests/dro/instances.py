"""Hand-picked problem instances shared by the ambiguity-set test modules.

Each instance is a regime that has broken an ambiguity set before or that
sits on a structural boundary of the worst case: a tied maximum, duplicate
support points, a zero-mass maximum, a wide loss spread, and a single
support point. `tests/dro/test_oracles.py` checks every set against an
independent reference on them, and `tests/dro/test_properties.py` replays
them as explicit Hypothesis examples, so a regression in any of these
regimes fails on every run rather than only when a random draw finds it.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True)
class Instance:
    """A nominal distribution, a loss vector, and a ground-cost geometry.

    Attributes:
        name: Identifier used as the pytest parameter id.
        nominal: Reference distribution over the support.
        loss: Per-scenario losses, one per support point.
        position: Coordinate of each support point on the real line, from
            which the Wasserstein ground cost is built as the pairwise
            absolute difference.
    """

    name: str
    nominal: tuple[float, ...]
    loss: tuple[float, ...]
    position: tuple[float, ...]

    @property
    def nominal_array(self) -> NDArray[np.float64]:
        """Return the nominal distribution as a float64 NumPy array."""
        return np.array(self.nominal, dtype=np.float64)

    @property
    def loss_array(self) -> NDArray[np.float64]:
        """Return the loss vector as a float64 NumPy array."""
        return np.array(self.loss, dtype=np.float64)

    @property
    def cost_array(self) -> NDArray[np.float64]:
        """Return the pairwise absolute-difference ground cost matrix."""
        position = np.array(self.position, dtype=np.float64)
        return np.abs(position[:, None] - position[None, :])

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(nominal, loss, cost)` as float64 tensors."""
        return (
            torch.tensor(self.nominal_array, dtype=torch.float64),
            torch.tensor(self.loss_array, dtype=torch.float64),
            torch.tensor(self.cost_array, dtype=torch.float64),
        )


INSTANCES = (
    Instance(
        name="asymmetric",
        nominal=(0.1, 0.2, 0.3, 0.4),
        loss=(0.0, 1.0, 2.0, 5.0),
        position=(0.0, 1.0, 2.0, 3.0),
    ),
    Instance(
        name="tied_maximum",
        nominal=(0.1, 0.2, 0.3, 0.4),
        loss=(0.0, 1.0, 5.0, 5.0),
        position=(0.0, 1.0, 2.0, 3.0),
    ),
    Instance(
        name="duplicate_support",
        nominal=(0.25, 0.25, 0.25, 0.25),
        loss=(1.0, 1.0, 3.0, 4.0),
        position=(0.0, 0.0, 1.0, 2.0),
    ),
    Instance(
        name="zero_weight_maximum",
        nominal=(0.0, 0.3, 0.3, 0.4),
        loss=(9.0, 1.0, 2.0, 5.0),
        position=(0.0, 1.0, 2.0, 3.0),
    ),
    Instance(
        name="wide_spread",
        nominal=(0.5, 0.5),
        loss=(0.0, 100.0),
        position=(0.0, 1.0),
    ),
    Instance(
        name="single_point",
        nominal=(1.0,),
        loss=(2.5,),
        position=(0.0,),
    ),
)
