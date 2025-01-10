from __future__ import annotations

from typing import Iterator, Self


class TrieTree:
    """Tree-like data structure for efficient reTRIEval of key-value pairs.

    It is an alternative to hash tables, with the advantage that it can be used
    for prefix-based word searching and sorted word traversal.

    Applications
    ------------
    Some examples include:
    - Lexicographic sorting
    - Autocomplete
    - Spell checkers

    Pros
    ----
    - Faster than hash tables and BSTs
    - Allows sorting
    - Efficient prefix matching

    Cons
    ----
    - Requires a lot of memory
    - Lookup is faster in an optimized hash table: O(1)
    """

    def __init__(self, alphabet_size: int = 26, first_character: str = "a") -> None:
        self._alphabet_size = alphabet_size
        self._first_character = first_character
        self._word_count = 0
        self._root = self._create_new_node()

    def __iter__(self) -> Iterator[str]:
        """Iterate through stored words in alphabetical order.

        Implements a preorder DFS.

        Complexity
        ----------
        Time: O(N)
        Space: O(N)
        """

        def preorder(root: _TrieNode, prefix: str = "") -> Iterator[tuple[_TrieNode, str]]:
            prefix += root.character
            if root.is_end_of_word:
                # A complete word has been found
                yield root, prefix
            for child in root:
                yield from preorder(child, prefix)

        for _, word in preorder(self._root):
            yield word

    def insert(self, word: str) -> None:
        """Insert a new word to the dictionary.

        Complexity
        ----------
        Time: O(word_size)
        Space: O(1)
        """
        assert word
        node = self._root
        for character in word.lower():
            if node[character] is None:
                node[character] = self._create_new_node(character)
            node = node[character]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        """Search if word exists in dictionary or not.

        Complexity
        ----------
        Time: O(word_size)
        Space: O(1)
        """
        assert word
        node = self._root
        for character in word.lower():
            if node[character] is None:
                return False
            node = node[character]
        return node.is_end_of_word

    def delete(self, word: str) -> None:
        """Delete word from dictionary.

        Complexity
        ----------
        Time: O(word_size)
        Space: O(1)
        """
        assert word

        # Find word node and last branch
        node = self._root
        last_branch_node_and_character: tuple[_TrieNode, str] | None = None
        for character in word.lower():
            assert node[character] is not None, f'Word "{word}" not found'
            if node.num_children > 0:
                if node.num_children > 1 or node.is_end_of_word:
                    # Node is a branch
                    last_branch_node_and_character = node, character
            node = node[character]

        if node.num_children > 0:
            # Deleted word is prefix of another word (not a leaf node)
            node.is_end_of_word = False
        elif last_branch_node_and_character is not None:
            # Deleted word shares a prefix with other words (delete branch)
            node, character = last_branch_node_and_character
            node[character] = None
        else:
            # Deleted word was the only word
            self._root[word[0]] = None

    def _create_new_node(self, value: str = "") -> _TrieNode:
        return _TrieNode(value, self._alphabet_size, self._first_character)


class _TrieNode:
    """Trie tree node.

    Its key is a single character, and may have multiple children.
    """

    def __init__(self, character: str, alphabet_size: int, first_character: str) -> None:
        self._character = character
        self._children: list[Self | None] = alphabet_size * [None]
        self._index_offset = ord(first_character)
        self.is_end_of_word = False

    @property
    def character(self) -> str:
        return self._character

    @property
    def num_children(self) -> int:
        return len(self._children) - self._children.count(None)

    def __iter__(self) -> Iterator[Self]:
        yield from (child for child in self._children if child is not None)

    def __getitem__(self, character: str) -> Self | None:
        index = self._get_index(character)
        return self._children[index]

    def __setitem__(self, character: str, node: Self) -> None:
        index = self._get_index(character)
        self._children[index] = node

    def _get_index(self, character: str) -> int:
        index = ord(character) - self._index_offset
        assert 0 <= index < len(self._children), f'Invalid character "{character}"'
        return index
