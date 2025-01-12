from __future__ import annotations

import abc
import math
from typing import Any, Iterator, Protocol, TypeAlias

import sympy

# Integer universe assumption (TODO: Add string support)
KeyType: TypeAlias = int
ValueType: TypeAlias = Any
KeyValuePairType: TypeAlias = tuple[KeyType, ValueType]


class HashTable(abc.ABC):
    """Associative array of key-value pairs, mapping keys to values.

    It uses a `hash_function` to transform keys into indices, which are then
    used to access the stored "buckets" in the underlying array. In a
    well-dimensioned hash table, the average complexity for all operations is
    O(1). In the worst-case when a hash collision occurs, the complexity becomes
    O(N).

    Hash Collisions
    ---------------
    Hash collision occurs when two or more keys are mapped to the same index.
    Different methods are used in order to address or avoid these situations.

    Applications
    ------------
    - Associative arrays (dictionary)
    - Database or search engine indexing
    - Caching
    - Cryptography
    """

    def __init__(self, *, hash_function: HashFunction, n_buckets: int = 16) -> None:
        assert n_buckets > 0
        self._array: list[KeyValuePairType | None] = n_buckets * [None]
        self._hash_function = hash_function

    @property
    def n_buckets(self) -> int:
        return len(self._array)

    @property
    @abc.abstractmethod
    def load_factor(self) -> float:
        """Critical statistic of hash table.

        It is defined as the number of entries over the total number of buckets.
        Large load factors increase the chance of hash collision, so they should
        be avoided by occasionally rehashing the table.
        """

    @abc.abstractmethod
    def __getitem__(self, key: KeyType) -> ValueType | None: ...

    @abc.abstractmethod
    def __setitem__(self, key: KeyType, value: ValueType | None) -> None: ...

    @abc.abstractmethod
    def items(self) -> Iterator[KeyValuePairType]: ...

    def _get_index(self, key: KeyType) -> int:
        return self._hash_function(key, self.n_buckets)

    def _rehash(self) -> None:
        """Double the total number of buckets."""
        keys_and_values = [(key, value) for key, value in self.items()]
        n_buckets_new = 2 * self.n_buckets
        self.__init__(hash_function=self._hash_function, n_buckets=n_buckets_new)
        for key, value in keys_and_values:
            self[key] = value


class SeparateChainingHashTable(HashTable):
    """Separate chain collision resolution technique.

    Stores a linked-list of key-value pairs for each bucket in the array, used
    to chain together pairs which are mapped to the same index.

    Comparison
    ----------
    + Suitable when the insertion/deletion rate is unknown
    + Simple
    + No chance of overflow
    + Less sensitive to hash function
    - Poor cache performance (no contiguous storage)
    - Wastes a lot of space (lots of unused buckets)
    - If chain becomes long, it can end up having O(N) complexity
    """

    def __init__(self, *, hash_function: HashFunction, n_buckets: int = 16) -> None:
        super().__init__(hash_function=hash_function, n_buckets=n_buckets)
        self._array: list[list[KeyValuePairType]] = [[] for _ in range(self.n_buckets)]

    @property
    def load_factor(self) -> float:
        n_entries = sum(len(entries) for entries in self._array)
        return n_entries / self.n_buckets

    def __getitem__(self, key: KeyType) -> ValueType | None:
        index = self._get_index(key)
        for key_stored, value_stored in self._array[index]:
            if key_stored == key:
                return value_stored
        return None

    def __setitem__(self, key: KeyType, value: ValueType | None) -> None:
        self._insert_value(key, value)
        if self.load_factor > 0.75:
            self._rehash()

    def items(self) -> Iterator[KeyValuePairType]:
        for bucket in self._array:
            yield from bucket

    def _insert_value(self, key: KeyType, value: ValueType | None) -> None:
        index = self._get_index(key)
        for subindex, (key_stored, _) in enumerate(self._array[index]):
            if key_stored != key:
                continue
            if value is None:
                self._array[index].pop(subindex)
            else:
                self._array[index][subindex] = (key, value)
            return
        self._array[index].append((key, value))


class _OpenAddressingHashTable(HashTable, abc.ABC):
    """Open addressing collision resolution technique.

    In case a key is mapped to a pre-existing index, it keeps probing (i.e.,
    exploring subsequent indices), until an unoccupied index is found.

    There are three main types of probing:
    - linear
    - quadratic
    - double hashing

    Comparison
    ----------
    + Suitable when the insertion/deletion rate is known
    + Good cache performance (contiguous storage)
    + Better space utilization
    - More computations
    - Table can overflow, and require rehashing
    - Extra care needed to avoid clustering
    """

    @property
    def load_factor(self) -> float:
        n_entries = sum(1.0 for entry in self._array if entry is not None)
        return n_entries / self.n_buckets

    def __getitem__(self, key: KeyType) -> ValueType | None:
        index_hashed = self._get_index(key)
        for n_attempt in range(self.n_buckets):
            index = self._compute_index(key, index_hashed, n_attempt)
            entry = self._array[index]
            if entry is not None and entry[0] == key:
                return entry[1]
        return None

    def __setitem__(self, key: KeyType, value: ValueType | None) -> None:
        try:
            self._insert_value(key, value)
        except AssertionError:
            self._rehash()
            self._insert_value(key, value)
        if self.load_factor > 0.75:
            self._rehash()

    def items(self) -> Iterator[KeyValuePairType]:
        for entry in self._array:
            if entry is not None:
                yield entry

    def _insert_value(self, key: KeyType, value: ValueType | None) -> None:
        index_start = self._get_index(key)
        for n_attempt in range(self.n_buckets):
            index = self._compute_index(key, index_start, n_attempt)
            entry = self._array[index]
            if entry is None or entry[0] == key:
                self._array[index] = (key, value) if value is not None else None
                return
        raise AssertionError

    @abc.abstractmethod
    def _compute_index(self, key: KeyType, index_hashed: int, n_attempt: int) -> int:
        """Find next candidate for search/insertion."""


class LinearProbingHashTable(_OpenAddressingHashTable):
    """Linear probing open addressing collision resolution technique.

    Sequentially go through all buckets, until an empty one is found.

    Comparison
    ----------
    + Simple
    + Good cache performance
    - Can lead to clustering at small values
    """

    def _compute_index(self, key: KeyType, index_hashed: int, n_attempt: int) -> int:
        return (index_hashed + n_attempt) % self.n_buckets


class QuadraticProbingHashTable(_OpenAddressingHashTable):
    """Quadratic probing open addressing collision resolution technique.

    Look at the n**2-th bucket in sequence at the n-th iteration.

    Comparison
    ----------
    + Good compromise between clustering and cache performance
    """

    def _compute_index(self, key: KeyType, index_hashed: int, n_attempt: int) -> int:
        return (index_hashed + n_attempt**2) % self.n_buckets


class DoubleHashingHashTable(_OpenAddressingHashTable):
    """Double hashing open addressing collision resolution technique.

    Use a second hash function to find next candidate for search or insertion.

    Comparison
    ----------
    + No clustering
    - Poor cache performance
    - Complex
    """

    def _compute_index(self, key: KeyType, index_hashed: int, n_attempt: int) -> int:
        return (index_hashed + n_attempt * self._hash_function_2(key)) % self.n_buckets

    def _hash_function_2(self, key: KeyType) -> int:
        # This is a popular second hashing function, see also here:
        # https://www.geeksforgeeks.org/double-hashing/
        prime = sympy.prevprime(self.n_buckets)
        return prime - (key % prime)


class HashFunction(Protocol):
    """Function that maps a key into an index.

    In general, a hash function is any function that maps data of arbitrary size
    to fixed-size values.

    A good hash function should be efficient and distribute keys uniformly, to
    minimize collisions.
    """

    def __call__(self, key: KeyType, n_buckets: int) -> int: ...


def division_method(key: KeyType, n_buckets: int) -> int:
    """Division hashing method.

    Applies a simple modulo function on the key, to ensure it produces an index
    within bounds. It has been found that the best results are obtained when a
    prime number close to `n_buckets` is used as the divisor.

    Comparison
    ----------
    + Simple
    - Clustering
    - Sensitive to table size
    """
    return key % n_buckets


def multiplication_method(key: KeyType, n_buckets: int) -> int:
    """Multiplication hashing method.

    Takes fractional part of the key multiplied by a constant, multiplies
    it by the bucket size, and then floors it.

    The constant is typically chosen as `s / 2**w`, where `w` is the machine
    word size, and `s` is a constant within `(0, 2**w)`.

    Comparison
    ----------
    + Simple
    + Insensitive to table size
    """
    word_size = 64
    fractional_part = (key * 1.0 / word_size**2) % 1
    return math.floor(n_buckets * fractional_part)


def midsquare_method(key: KeyType, n_buckets: int) -> int:
    """Mid-square hashing method.

    Takes the middle digits of the squared input integer.

    Comparison
    ----------
    + Reasonable hash code
    - More complex
    - Poor performance in case of many trailing or leading zeros
    """
    key_squared_str = str(key**2)
    n_digits_bucket = len(str(n_buckets))
    n_digits_key_squared = len(key_squared_str)
    if n_digits_key_squared < n_digits_bucket:
        pad_size = n_digits_bucket - n_digits_key_squared
        key_squared_str = key_squared_str.zfill(pad_size)
    index_middle = len(key_squared_str) // 2
    middle_digits = int(key_squared_str[index_middle - n_buckets // 2 : index_middle + n_buckets // 2])
    return middle_digits % n_buckets
