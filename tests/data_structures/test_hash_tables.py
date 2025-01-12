from __future__ import annotations

import pytest
from pytest import approx

from dsa.data_structures import hash_tables

table_types = [
    hash_tables.SeparateChainingHashTable,
    hash_tables.LinearProbingHashTable,
    hash_tables.QuadraticProbingHashTable,
    hash_tables.DoubleHashingHashTable,
]
hash_functions = [hash_tables.division_method, hash_tables.multiplication_method, hash_tables.midsquare_method]


@pytest.mark.parametrize("hash_function", hash_functions, ids=lambda c: c.__name__)
@pytest.mark.parametrize("table_type", table_types, ids=lambda c: c.__name__)
def test_hash_table_insert_one(table_type: type[hash_tables.HashTable], hash_function: hash_tables.HashFunction):
    table = table_type(hash_function=hash_function, n_buckets=8)
    assert table[42] is None
    assert table.load_factor == approx(0.0)

    table[42] = "42"

    assert table[42] == "42"
    assert table.load_factor == approx(1.0 / 8.0)


@pytest.mark.parametrize("hash_function", hash_functions, ids=lambda c: c.__name__)
@pytest.mark.parametrize("table_type", table_types, ids=lambda c: c.__name__)
def test_hash_table_remove_one(table_type: type[hash_tables.HashTable], hash_function: hash_tables.HashFunction):
    table = table_type(hash_function=hash_function, n_buckets=8)
    table[420] = "420"
    assert table[420] == "420"
    assert table.load_factor == approx(1.0 / 8.0)

    table[420] = None

    assert table[420] is None
    assert table.load_factor == approx(0.0)


@pytest.mark.parametrize("n_buckets", [4, 8, 16, 32], ids=lambda s: f"{s}_bucket")
@pytest.mark.parametrize("hash_function", hash_functions, ids=lambda c: c.__name__)
@pytest.mark.parametrize("table_type", table_types, ids=lambda c: c.__name__)
def test_hash_table_handle_collisions(
    table_type: type[hash_tables.HashTable], hash_function: hash_tables.HashFunction, n_buckets: int
):
    table = table_type(hash_function=hash_function, n_buckets=n_buckets)
    for key in range(2, 40, 2):
        table[key] = str(key)
        assert table.load_factor <= 0.75

    for key in range(2, 40, 2):
        assert table[key] == str(key)
