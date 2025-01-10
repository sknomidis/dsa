from __future__ import annotations

import pytest

from dsa.data_structures import trees


def test_trie_insert() -> None:
    tree = trees.TrieTree()
    assert not tree.search("and")
    tree.insert("and")
    assert tree.search("and")
    assert not tree.search("ant")
    tree.insert("ant")
    assert tree.search("ant")
    assert not tree.search("an")


@pytest.mark.parametrize(
    "inserted, deleted",
    [
        (["and"], ["and"]),
        (["and", "ant", "antihero"], ["and"]),
        (["and", "ant", "antihero"], ["ant"]),
        (["and", "ant", "bat"], ["bat"]),
    ],
    ids=["only_word", "shared_prefix", "is_prefix", "no_shared_prefix"],
)
def test_trie_delete(inserted: list[str], deleted: list[str]) -> None:
    tree = trees.TrieTree()
    for word in inserted:
        tree.insert(word)
    for word in inserted:
        assert tree.search(word)
    for word in deleted:
        tree.delete(word)
    for word in inserted:
        if word in deleted:
            assert not tree.search(word)
        else:
            assert tree.search(word)
    with pytest.raises(AssertionError):
        tree.delete("invalid")


@pytest.mark.parametrize(
    "words",
    [[], ["foo"], ["foo", "bar"], ["foo", "bar", "baz"], ["baz", "foo", "foobar", "bar"]],
    ids=lambda w: f"{len(w)}_word(s)",
)
def test_trie_list_words_sorted(words: list[str]) -> None:
    tree = trees.TrieTree()
    for word in words:
        tree.insert(word)
    assert list(tree) == sorted(words)
