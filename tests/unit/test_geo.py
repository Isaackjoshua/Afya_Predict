"""Geospatial helpers and the mobility fallbacks."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.geo import (
    distance_matrix,
    gravity_matrix,
    haversine_km,
    neighbours,
    population_series,
    radiation_matrix,
    subset_region,
)


def test_haversine_matches_a_known_distance():
    # Dar es Salaam -> Mwanza is roughly 870 km great-circle.
    km = haversine_km(-6.79, 39.21, -2.52, 32.90)
    assert 800 < km < 950


def test_distance_matrix_is_symmetric_with_zero_diagonal(small_region):
    matrix = distance_matrix(small_region)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
    assert np.allclose(np.diag(matrix.to_numpy()), 0.0)


@pytest.mark.parametrize("builder", [gravity_matrix, radiation_matrix])
def test_flow_matrices_are_row_normalised_with_no_self_flow(small_region, builder):
    matrix = builder(small_region)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert np.allclose(np.diag(matrix.to_numpy()), 0.0)
    assert (matrix.to_numpy() >= 0).all()


def test_gravity_flow_prefers_near_and_large_districts(region):
    row = gravity_matrix(region).loc["Kinondoni"]
    # Ilala is adjacent and populous; Songea is 800 km away.
    assert row["Ilala"] > row["Songea MC"] * 50


def test_neighbours_excludes_self_and_respects_k(region):
    result = neighbours(region, "Mwanza City", k=3)
    assert len(result) == 3
    assert "Mwanza City" not in result


def test_subset_region_keeps_only_named_districts(region):
    subset = subset_region(region, ["Kinondoni", "Ilala"])
    assert subset.district_names == ["Kinondoni", "Ilala"]
    assert population_series(subset).sum() > 0
