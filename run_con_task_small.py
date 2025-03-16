import torch
import time, json, glob, gc, os

from typing import List, Union
from phrasetree.tree import Tree
from transformers import AutoConfig, AutoModel, AutoTokenizer
from src.stl_model.con_model import CRFConstituencyModel, CRFConstituencyDecoder
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
        'token_length': torch.tensor([data['token_length'] for data in batch_data], device=device),
        'token_': [[data['token_']] for data in batch_data],
    }

    # Pad token_input_ids
    padded_input_ids = []
    for data in batch_data:
        pad_len = max_input_ids_len - len(data['token_input_ids'])
        padded = data['token_input_ids'] + [0] * pad_len
        padded_input_ids.append(torch.tensor(padded, device=device))
    batched['token_input_ids'] = torch.stack(padded_input_ids)


    # Pad token_token_span
    padded_spans = []
    for data in batch_data:
        curr_spans = data['token_token_span']
        curr_spans = [span + [0] * (max_span_width - len(span)) for span in curr_spans]
        pad_len = max_token_span_len - len(curr_spans)
        padded = curr_spans + [[0] * max_span_width] * pad_len
        padded_spans.append(torch.tensor(padded, device=device))
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
    out_path = inp_path.replace('.ele_b.tok', '.ele_s.con')

    if os.path.exists(out_path):
        print(f'{progress} Skipping {inp_path}')
        return False

    print(f'{progress} Processing {inp_path}')

    with open(inp_path, 'r') as f:
        lines = f.readlines()
        sentences = [line.strip().split('\t') for line in lines]

    if len(sentences) == 0:
        return False

    results = process_sentences(sentences)
    outputs = [tree_to_inline(tree) for tree in results]

    # Clear GPU cache after each batch
    torch.cuda.empty_cache()

    with open(out_path, 'w') as f:
        f.write('\n'.join(outputs))

    return True


with open("./config/con/vocabs-electra.json", "r", encoding="utf-8") as f:
    VOCAB = json.load(f)["chart"]["idx_to_token"]



config_trans = AutoConfig.from_pretrained("./config/con/config-electra.json")
pre_model = AutoModel.from_config(config=config_trans)
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/electra")

encoder = ContextualWordEmbeddingModule("token", transformer=pre_model, transformer_tokenizer=tokenizer, word_dropout=.1)
decoder = CRFConstituencyDecoder(n_labels=260, n_hidden=256, n_mlp_span=500, n_mlp_label=100, mlp_dropout=.33)

transform = TransformerSequenceTokenizer("./tokenizer/electra", 'token', cls_is_bos=True, truncate_long_sequences=False)

device = torch.device('cuda')

model = CRFConstituencyModel(encoder, decoder)
model.load_state_dict(torch.load("./model/con-ctb9-electra-small.pt", map_location=device, weights_only=True), strict=False)
model.to(device)  # Move model to GPU if available
model.eval()


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
