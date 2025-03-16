import hanlp, torch

from typing import List, Union
from phrasetree.tree import Tree

def tree_to_inline(tree):
    """Convert a Tree object to a single-line string representation."""
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1 and not isinstance(tree[0], Tree):
        return f"({tree.label()} {tree[0]})"
    else:
        children = ' '.join(tree_to_inline(child) for child in tree)
        return f"({tree.label()} {children})"

con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ELECTRA_SMALL)

tree = con(['石天道', ':'])
print(tree_to_inline(tree))
