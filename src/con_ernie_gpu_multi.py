import json
import torch
import glob, gc, os

from typing import List, Union
from phrasetree.tree import Tree
from transformers import AutoConfig, AutoModel, AutoTokenizer
from stl_model.con_model import CRFConstituencyModel, CRFConstituencyDecoder
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp.layers.embeddings.contextual_word_embedding import ContextualWordEmbeddingModule


def build_tree(tokens: List[str], sequence):
    if not tokens:
        return Tree('TOP', [])

    tree = Tree('TOP', [Tree('_', [t]) for t in tokens])
    root = tree.label()
    leaves = [subtree for subtree in tree.subtrees() if not isinstance(subtree[0], Tree)]

    def track(node):
        i, j, label = next(node)
        if j == i + 1:
            children = [leaves[i]]
        else:
            children = track(node) + track(node)

        if label.endswith('|<>'):
            return children

        labels = label.split('+')
        tree = Tree(labels[-1], children)
        for label in reversed(labels[:-1]):
            tree = Tree(label, [tree])
        return [tree]

    return Tree(root, track(iter(sequence)))

def tree_to_inline(tree):
    """Convert a Tree object to a single-line string representation."""
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1 and not isinstance(tree[0], Tree):
        return f"({tree.label()} {tree[0]})"
    else:
        children = ' '.join(tree_to_inline(child) for child in tree)
        return f"({tree.label()} {children})"

with open("./config/con/vocabs-ernie.json", "r", encoding="utf-8") as f:
    VOCAB = json.load(f)["chart"]["idx_to_token"]


def process_sentences(sentences: List[List[str]]):
    """Process multiple sentences and return their constituency trees.

    Args:
        sentences: A list of tokenized sentences, each being a list of tokens

    Returns:
        A list of constituency parse trees
    """

    # Process each sentence and collect batches
    batch_data = []
    for sentence in sentences:
        batch = transform({"token": sentence})
        token_token_span = batch['token_token_span']
        max_len = max(len(span) for span in token_token_span)
        padded_spans = [span + [0] * (max_len - len(span)) for span in token_token_span]
        batch['token_token_span'] = padded_spans

        processed_batch = {
            'token': batch['token'],
            '_idx_': 0,
            'token_length': len(batch['token']),
            'token_': [token.capitalize() if token.isalpha() else token for token in batch['token']],
            'token_input_ids': batch['token_input_ids'],
            'token_token_span': batch['token_token_span']
        }
        batch_data.append(processed_batch)

    # Find maximum lengths for padding
    max_input_ids_len = max(len(data['token_input_ids']) for data in batch_data)
    max_token_span_len = max(len(data['token_token_span']) for data in batch_data)
    max_span_width = max(len(span) for data in batch_data for span in data['token_token_span'])

    # Create batch tensors with proper padding
    batched = {
        'token': [data['token'] for data in batch_data],
        '_idx_': list(range(len(batch_data))),
        'token_length': torch.tensor([data['token_length'] for data in batch_data]),
        'token_': [[data['token_']] for data in batch_data],
    }

    # Pad token_input_ids
    padded_input_ids = []
    for data in batch_data:
        pad_len = max_input_ids_len - len(data['token_input_ids'])
        padded = data['token_input_ids'] + [0] * pad_len  # Use tokenizer.pad_token_id instead of 0 if available
        padded_input_ids.append(torch.tensor(padded))
    batched['token_input_ids'] = torch.stack(padded_input_ids)

    # Pad token_token_span - needs to be padded in two dimensions
    padded_spans = []
    for data in batch_data:
        curr_spans = data['token_token_span']
        # First ensure all spans have the same width
        curr_spans = [span + [0] * (max_span_width - len(span)) for span in curr_spans]
        # Then pad to the same length
        pad_len = max_token_span_len - len(curr_spans)
        padded = curr_spans + [[0] * max_span_width] * pad_len
        padded_spans.append(torch.tensor(padded))
    batched['token_token_span'] = torch.stack(padded_spans)

    # Run model
    s_span, s_label = model(batched)

    # Process results
    tokens = [batch_data[i]['token'] for i in range(len(batch_data))]

    offset = 1
    lens = batched['token_length'] - offset
    seq_len = lens.max()
    mask = lens.new_tensor(range(seq_len)) < lens.view(-1, 1, 1)
    mask = mask & mask.new_ones(seq_len, seq_len).triu_(1)

    if mask.any().item():
        s_span = decoder.crf(s_span, mask, mbr=True)

    chart_preds = decoder.decode(s_span, s_label, mask)

    result = [build_tree(token, [(i, j, VOCAB[label]) for i, j, label in chart])
              for token, chart in zip(tokens, chart_preds)]

    return result


config_trans = AutoConfig.from_pretrained('./config/con/config-ernie.json')
tokenizer = AutoTokenizer.from_pretrained('./tokenizer/ernie')
pre_model = AutoModel.from_config(config=config_trans)
encoder = ContextualWordEmbeddingModule("token", transformer=pre_model, transformer_tokenizer=tokenizer, word_dropout=.1)
decoder = CRFConstituencyDecoder(n_labels=1353, n_hidden=768, n_mlp_span=500, n_mlp_label=100, mlp_dropout=.33)

transform = TransformerSequenceTokenizer('./tokenizer/ernie', 'token', cls_is_bos=True, truncate_long_sequences=False)

model = CRFConstituencyModel(encoder, decoder)
model.load_state_dict(torch.load( './model/con-ctb9-ernie-gram.pt', map_location='cuda', weights_only=True), strict=False)
model.eval()


with open('/data/hanlp/uaa/852403603686690816-198.ele_b.tok', 'r') as f:
    lines = f.readlines()
    sentences = [line.strip().split('\t') for line in lines]

results = process_sentences(sentences)
print("\nMultiple sentences results:")
for i, tree in enumerate(results):
    print(f"Sentence {i+1}:")
    print(tree_to_inline(tree))
    # print(tree.pretty_print())
