#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_evaluation.py
==================
TDD unit tests for the *pure* metric functions in ``evaluation.py``.

These tests deliberately depend ONLY on numpy / scipy (and ``evaluation`` whose
heavy deps are lazy-imported), so they run on a CPU-only host without
TensorFlow / a GPU.

Run with:
    pytest -v test_evaluation.py
"""

import numpy as np
import pytest

from evaluation import (
    extract_boundary,
    boundary_f1_score,
    hausdorff_distance,
    mask_iou,
    extract_map_at_iou,
)


# ----------------------------------------------------------------------------
# Mock-data helpers
# ----------------------------------------------------------------------------
def square_mask(size=64, top=10, left=10, h=20, w=20):
    """A filled rectangle on an otherwise empty ``size x size`` canvas."""
    m = np.zeros((size, size), dtype=bool)
    m[top:top + h, left:left + w] = True
    return m


def circle_mask(size=64, cy=32, cx=32, r=12):
    yy, xx = np.ogrid[:size, :size]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


# ============================================================================
# extract_boundary
# ============================================================================
def test_extract_boundary_is_hollow():
    """The boundary must be non-empty but strictly smaller than the filled area."""
    m = square_mask(h=20, w=20)
    b = extract_boundary(m)
    assert b.any()
    assert b.sum() < m.sum()                 # interior pixels removed
    assert np.all(b <= m)                     # boundary is a subset of the mask


def test_extract_boundary_empty_mask():
    m = np.zeros((32, 32), dtype=bool)
    assert extract_boundary(m).sum() == 0


def test_extract_boundary_single_pixel():
    m = np.zeros((16, 16), dtype=bool)
    m[5, 5] = True
    # a single pixel has no interior to erode -> it IS its own boundary
    assert extract_boundary(m).sum() == 1


# ============================================================================
# Boundary F1-score
# ============================================================================
def test_bf1_identical_masks_is_one():
    m = circle_mask()
    assert boundary_f1_score(m, m) == pytest.approx(1.0)


def test_bf1_disjoint_masks_is_zero():
    a = square_mask(size=64, top=2, left=2, h=10, w=10)
    b = square_mask(size=64, top=50, left=50, h=10, w=10)   # far away
    assert boundary_f1_score(a, b, tolerance=2.0) == pytest.approx(0.0)


def test_bf1_both_empty_is_one():
    z = np.zeros((32, 32), dtype=bool)
    assert boundary_f1_score(z, z) == pytest.approx(1.0)


def test_bf1_one_empty_is_zero():
    z = np.zeros((32, 32), dtype=bool)
    m = square_mask(size=32, top=5, left=5, h=8, w=8)
    assert boundary_f1_score(m, z) == pytest.approx(0.0)
    assert boundary_f1_score(z, m) == pytest.approx(0.0)


def test_bf1_partial_overlap_between_0_and_1():
    """A small shift should give an intermediate score with a tight tolerance."""
    a = square_mask(size=80, top=20, left=20, h=30, w=30)
    b = square_mask(size=80, top=20, left=26, h=30, w=30)   # shifted by 6 px
    score = boundary_f1_score(a, b, tolerance=1.0)
    assert 0.0 < score < 1.0


def test_bf1_is_symmetric():
    a = circle_mask(cy=30, cx=30, r=12)
    b = circle_mask(cy=33, cx=31, r=12)
    assert boundary_f1_score(a, b) == pytest.approx(boundary_f1_score(b, a))


def test_bf1_tolerance_is_monotonic():
    """A looser tolerance can only keep or increase the score."""
    a = circle_mask(cy=30, cx=30, r=12)
    b = circle_mask(cy=33, cx=31, r=12)
    assert boundary_f1_score(a, b, tolerance=3.0) >= boundary_f1_score(a, b, tolerance=1.0)


# ============================================================================
# Hausdorff Distance
# ============================================================================
def test_hd_identical_masks_is_zero():
    m = circle_mask()
    assert hausdorff_distance(m, m) == pytest.approx(0.0)


def test_hd_both_empty_is_zero():
    z = np.zeros((32, 32), dtype=bool)
    assert hausdorff_distance(z, z) == pytest.approx(0.0)


def test_hd_one_empty_is_inf():
    z = np.zeros((32, 32), dtype=bool)
    m = square_mask(size=32, top=5, left=5, h=8, w=8)
    assert hausdorff_distance(m, z) == float("inf")
    assert hausdorff_distance(z, m) == float("inf")


def test_hd_disjoint_is_large_and_matches_scipy():
    """Cross-check our HD against scipy's directed_hausdorff on boundary points."""
    from scipy.spatial.distance import directed_hausdorff
    a = square_mask(size=64, top=2, left=2, h=10, w=10)
    b = square_mask(size=64, top=50, left=50, h=10, w=10)
    hd = hausdorff_distance(a, b)

    pa = np.argwhere(extract_boundary(a))
    pb = np.argwhere(extract_boundary(b))
    expected = max(directed_hausdorff(pa, pb)[0], directed_hausdorff(pb, pa)[0])

    assert hd == pytest.approx(expected, abs=1e-6)
    assert hd > 40.0                           # corners are far apart


def test_hd_translation_increases_with_shift():
    """Larger translations must produce a (weakly) larger Hausdorff distance."""
    base = square_mask(size=128, top=40, left=40, h=30, w=30)
    near = square_mask(size=128, top=40, left=45, h=30, w=30)   # +5 px
    far = square_mask(size=128, top=40, left=60, h=30, w=30)    # +20 px
    assert hausdorff_distance(base, far) > hausdorff_distance(base, near) > 0.0


def test_hd_translation_known_value():
    """Translating a shape by k px puts its far edge k px from the original."""
    base = square_mask(size=128, top=40, left=40, h=30, w=30)
    shifted = square_mask(size=128, top=40, left=48, h=30, w=30)   # +8 px in x
    hd = hausdorff_distance(base, shifted)
    assert hd == pytest.approx(8.0, abs=1.0)


def test_hd95_not_greater_than_max_hd():
    """The 95th-percentile HD is a robust lower-or-equal bound of the max HD."""
    a = circle_mask(cy=40, cx=40, r=15)
    b = circle_mask(cy=44, cx=43, r=15)
    hd = hausdorff_distance(a, b, percentile=None)
    hd95 = hausdorff_distance(a, b, percentile=95)
    assert hd95 <= hd + 1e-9


# ============================================================================
# mask_iou (used by the failure-case logic)
# ============================================================================
def test_iou_identical_is_one():
    m = circle_mask()
    assert mask_iou(m, m) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = square_mask(size=64, top=2, left=2, h=10, w=10)
    b = square_mask(size=64, top=50, left=50, h=10, w=10)
    assert mask_iou(a, b) == pytest.approx(0.0)


def test_iou_half_overlap():
    a = np.zeros((10, 10), dtype=bool); a[:, :6] = True       # 60 px
    b = np.zeros((10, 10), dtype=bool); b[:, 4:] = True       # 60 px
    # intersection cols 4,5 -> 20 px; union -> 100 px
    assert mask_iou(a, b) == pytest.approx(20.0 / 100.0)


# ============================================================================
# Input validation
# ============================================================================
def test_shape_mismatch_raises():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 12), dtype=bool)
    with pytest.raises(ValueError):
        boundary_f1_score(a, b)
    with pytest.raises(ValueError):
        hausdorff_distance(a, b)


def test_non_2d_raises():
    with pytest.raises(ValueError):
        extract_boundary(np.zeros((4, 4, 3), dtype=bool))


# ============================================================================
# COCO mAP extraction (pure-numpy logic, fake COCOeval object)
# ============================================================================
class _FakeParams:
    def __init__(self, iou_thrs):
        self.iouThrs = np.asarray(iou_thrs)


class _FakeCocoEval:
    """Mimics the attributes ``extract_map_at_iou`` touches."""
    def __init__(self, iou_thrs, precision):
        self.params = _FakeParams(iou_thrs)
        self.eval = {"precision": np.asarray(precision)}


def _make_fake_eval():
    iou_thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)      # 10 thresholds
    T = len(iou_thrs)
    R, K, A, M = 101, 2, 4, 3                              # recall, cats, areas, maxdets
    precision = np.full((T, R, K, A, M), -1.0)
    # Give threshold 0.75 (index 5) a constant precision of 0.8,
    # and threshold 0.90 (index 8) a constant precision of 0.4.
    precision[5, :, :, 0, -1] = 0.8
    precision[8, :, :, 0, -1] = 0.4
    return _FakeCocoEval(iou_thrs, precision)


def test_extract_map_at_iou_75():
    ce = _make_fake_eval()
    assert extract_map_at_iou(ce, 0.75) == pytest.approx(0.8)


def test_extract_map_at_iou_90():
    ce = _make_fake_eval()
    assert extract_map_at_iou(ce, 0.90) == pytest.approx(0.4)


def test_extract_map_at_iou_all_invalid_is_nan():
    iou_thrs = np.round(np.arange(0.5, 1.0, 0.05), 2)
    precision = np.full((len(iou_thrs), 101, 2, 4, 3), -1.0)   # nothing valid
    ce = _FakeCocoEval(iou_thrs, precision)
    assert np.isnan(extract_map_at_iou(ce, 0.5))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
