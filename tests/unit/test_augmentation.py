"""Augmentation tests."""

import torch

from digit_classifier.training.datasets import random_shift


def make_batch() -> torch.Tensor:
    x = torch.zeros(8, 1, 28, 28)
    x[:, :, 10:18, 10:18] = 1.0
    return x


def test_zero_shift_is_identity() -> None:
    x = make_batch()
    g = torch.Generator().manual_seed(0)
    assert torch.equal(random_shift(x, 0, g), x)


def test_shape_dtype_and_mass_preserved() -> None:
    x = make_batch()
    g = torch.Generator().manual_seed(1)
    y = random_shift(x, 2, g)
    assert y.shape == x.shape and y.dtype == x.dtype
    # roll moves pixels, never creates or destroys them
    assert torch.allclose(y.sum(), x.sum())


def test_deterministic_given_seed() -> None:
    x = make_batch()
    a = random_shift(x, 2, torch.Generator().manual_seed(7))
    b = random_shift(x, 2, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_shift_bounded() -> None:
    x = make_batch()
    y = random_shift(x, 2, torch.Generator().manual_seed(3))
    # original block occupies rows/cols 10..17; after +/-2 shift it must stay
    # within 8..19 in both axes
    nz = y.nonzero()
    assert nz[:, 2].min() >= 8 and nz[:, 2].max() <= 19
    assert nz[:, 3].min() >= 8 and nz[:, 3].max() <= 19
