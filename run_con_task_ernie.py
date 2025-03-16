import hanlp, torch
import glob, gc, os

from typing import List, Union
from phrasetree.tree import Tree

CON_TASK = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM)


def tree_to_inline(tree):
    """Convert a Tree object to a single-line string representation."""
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1 and not isinstance(tree[0], Tree):
        return f"({tree.label()} {tree[0]})"
    else:
        children = ' '.join(tree_to_inline(child) for child in tree)
        return f"({tree.label()} {children})"

def parse_file(inp_path, progress=''):
    out_path = inp_path.replace('.ele_b.tok', '.ernie.con')

    if os.path.exists(out_path):
        print(f'{progress} Skipping {inp_path}')
        return False

    print(f'{progress} Processing {inp_path}')

    with open(inp_path, 'r') as f:
        lines = f.readlines()
        sentences = [line.strip().split('\t') for line in lines]

    if len(sentences) == 0:
        return False

    results = CON_TASK(sentences)
    outputs = [tree_to_inline(tree) for tree in results]

    with open(out_path, 'w') as f:
        f.write('\n'.join(outputs))
    torch.cuda.empty_cache()

    return True


file_paths = glob.glob('/data/hanlp/*/*.ele_b.tok')
file_count = len(file_paths)


for index, file_path in enumerate(file_paths):
    try:
        fresh = parse_file(file_path, f'- <{index + 1}/{file_count}>')
        gc.collect()

    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        torch.cuda.empty_cache()
        gc.collect()
        continue
