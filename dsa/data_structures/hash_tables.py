from __future__ import annotations

import abc
import math
from typing import Iterator, Protocol, TypeAlias, TypeVar

# Integer universe assumption (TODO: String support)
KeyType: TypeAlias = int
ValueType = TypeVar("ValueType")
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

    def __init__(self, *, hash_function: HashFunction, num_buckets: int = 32) -> None:
        assert num_buckets > 0
        self._array: list[KeyValuePairType | None] = num_buckets * [None]
        self._hash_function = hash_function

    @property
    def num_buckets(self) -> int:
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

    def keys(self) -> Iterator[KeyType]:
        for key, _ in self.items():
            yield key

    def values(self) -> Iterator[ValueType]:
        for _, value in self.items():
            yield value

    def _get_index(self, key: KeyType) -> int:
        return self._hash_function(key, self.num_buckets)

    def _rehash(self) -> None:
        """Double the total number of buckets."""
        keys_and_values = list(self.items())
        self.__init__(hash_function=self._hash_function, num_buckets=2 * self.num_buckets)
        for key, value in keys_and_values:
            self[key] = value


class SeparateChainingHashTable(HashTable):
    """Separate chain collision resolution technique.

    Stores a linked-list of key-value pairs for each bucket in the array, used
    to chain together pairs which are mapped to the same index.

    Pros
    ----
    - Suitable when the insertion/deletion rate is unknown
    - Simple
    - No chance of overflow
    - Less sensitive to hash function

    Cons
    ----
    - Poor cache performance (no contiguous storage)
    - Wastes a lot of space (lots of unused buckets)
    - If chain becomes long, it can end up having O(N) complexity
    """

    def __init__(self, *, hash_function: HashFunction, num_buckets: int = 16) -> None:
        super().__init__(hash_function=hash_function, num_buckets=num_buckets)
        self._array: list[list[KeyValuePairType]] = [[] for _ in range(self.num_buckets)]

    @property
    def load_factor(self) -> float:
        num_entries = sum(len(entries) for entries in self._array)
        return num_entries / self.num_buckets

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

    Pros
    ----
    - Suitable when the insertion/deletion rate is known
    - Good cache performance (contiguous storage)
    - Better space utilization

    Cons
    ----
    - More computations
    - Table can overflow, and require rehashing
    - Extra care needed to avoid clustering
    """

    @property
    def load_factor(self) -> float:
        num_entries = len(self._array) - self._array.count(None)
        return num_entries / self.num_buckets

    def __getitem__(self, key: KeyType) -> ValueType | None:
        index_hashed = self._get_index(key)
        for num_attempt in range(self.num_buckets):
            index = self._compute_index(key, index_hashed, num_attempt)
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
        for num_attempt in range(self.num_buckets):
            index = self._compute_index(key, index_start, num_attempt)
            entry = self._array[index]
            if entry is None or entry[0] == key:
                self._array[index] = (key, value) if value is not None else None
                return
        raise AssertionError

    @abc.abstractmethod
    def _compute_index(self, key: KeyType, index_hashed: int, num_attempt: int) -> int:
        """Find next candidate for search/insertion."""


class LinearProbingHashTable(_OpenAddressingHashTable):
    """Linear probing open addressing collision resolution technique.

    Sequentially go through all buckets, until an empty one is found.

    Pros
    ----
    - Simple
    - Good cache performance

    Cons
    ----
    - Can lead to clustering at small values
    """

    def _compute_index(self, key: KeyType, index_hashed: int, num_attempt: int) -> int:
        return (index_hashed + num_attempt) % self.num_buckets


class QuadraticProbingHashTable(_OpenAddressingHashTable):
    """Quadratic probing open addressing collision resolution technique.

    Look at the n**2-th bucket in sequence at the n-th iteration. This method
    offers a good compromise between clustering prevention and cache performance.
    """

    def _compute_index(self, key: KeyType, index_hashed: int, num_attempt: int) -> int:
        return (index_hashed + num_attempt**2) % self.num_buckets


class DoubleHashingHashTable(_OpenAddressingHashTable):
    """Double hashing open addressing collision resolution technique.

    Use a second hash function to find next candidate for search or insertion.

    Pros
    ----
    - No clustering

    Cons
    ----
    - Poor cache performance
    """

    def _compute_index(self, key: KeyType, index_hashed: int, num_attempt: int) -> int:
        return (index_hashed + num_attempt * self._hash_function_2(key)) % self.num_buckets

    def _hash_function_2(self, key: KeyType) -> int:
        prime = 11
        return prime - (key % prime)


class HashFunction(Protocol):
    """Function that maps a key into an index.

    In general, a hash function is any function that maps data of arbitrary size
    to fixed-size values.

    A good hash function should be efficient and distribute keys uniformly, to
    minimize collisions.
    """

    def __call__(self, key: KeyType, num_buckets: int) -> int: ...


def division_method(key: KeyType, num_buckets: int) -> int:
    """Division hashing method.

    Applies a simple modulo function on the key, to ensure it produces an index
    within bounds. It has been found that the best results are obtained when a
    prime number close to `num_buckets` is used as the divisor.

    Pros
    ----
    - Simple

    Cons
    ----
    - Clustering
    - Sensitive to table size
    """
    return key % num_buckets


def multiplication_method(key: KeyType, num_buckets: int) -> int:
    """Multiplication hashing method.

    Takes fractional part of the key multiplied by a constant, multiplies
    it by the bucket size, and then floors it.

    The constant is typically chosen as `s / 2**w`, where `w` is the machine
    word size, and `s` is a constant within `(0, 2**w)`.

    Pros
    ----
    - Simple
    - Insensitive to table size
    """
    word_size = 64
    fractional_part = (key * 1.0 / word_size**2) % 1
    return math.floor(num_buckets * fractional_part)


def midsquare_method(key: KeyType, num_buckets: int) -> int:
    """Mid-square hashing method.

    Takes the middle digits of the squared input integer.

    Pros
    ----
    - Reasonable hash code

    Cons
    ----
    - More complex
    - Poor performance in case of many trailing or leading zeros
    """
    key_squared_str = str(key**2)
    num_digits_bucket = len(str(num_buckets))
    num_digits_key_squared = len(key_squared_str)
    if num_digits_key_squared < num_digits_bucket:
        pad_size = num_digits_bucket - num_digits_key_squared
        key_squared_str = key_squared_str.zfill(pad_size)
    index_middle = len(key_squared_str) // 2
    middle_digits = int(key_squared_str[index_middle - num_buckets // 2 : index_middle + num_buckets // 2])
    return middle_digits % num_buckets
