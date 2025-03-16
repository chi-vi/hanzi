import onnxruntime as ort
import numpy as np
import time, json, glob, gc, os
from concurrent.futures import ThreadPoolExecutor
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
import torch  # Add torch import

OPTIONS = ort.SessionOptions()
OPTIONS.intra_op_num_threads=6
OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

OPTIONS.enable_mem_pattern = True
OPTIONS.enable_mem_reuse = True

# Set the log severity level to 1 to increase the verbosity of the logging output
# OPTIONS.log_severity_level = 1

SESSION = ort.InferenceSession("./model/ner-msra-electra-small.onnx", OPTIONS, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) #

tokenizer = TransformerSequenceTokenizer("./tokenizer/electra", 'token', ret_subtokens=False, ret_subtokens_group=False, ret_token_span=True, use_fast=True)
# tokenizer = TransformerSequenceTokenizer("./tokenizer/electra", 'token', ret_token_span=True, ret_subtokens=False, ret_mask_and_type=False, ret_subtokens_group=False, ret_prefix_mask=False,)

with open("./config/ner/vocabs.json", "r") as f:
    VOCABS = json.load(f)["tag"]["idx_to_token"]

def decode_output(logits, mask, batch):
    if isinstance(logits, torch.Tensor):
        logits = logits.tolist()

    predictions = []
    for sent_logits in logits:
        tags = [VOCABS[idx] for idx in sent_logits]
        predictions.append(tags)

    spans = []
    sents = [sent['token'] for sent in batch]

    for tags, tokens in zip(predictions, sents):
        entities = get_entities(tags)
        spans.append(entities)
    return spans

def get_entities(tags):
    entities = []
    label = None
    start = None
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            if label:
                entities.append([label, start, i])
            label = tag[2:]
            start = i
            if tag.startswith('M-'):
                if not label:
                    continue
            if tag.startswith('E-'):
                if label:
                    entities.append((label, start, i + 1))
                label = None
                start = None
            else:
                entities.append((label, start, i+1))

        elif tag.startswith('S-'):
            entities.append((tag[2:], i, i + 1))
            label = None
            start = None
        elif tag == 'O':
            continue
    return entities

def process_sentences(sentences):
    # First, collect all data and determine max dimensions
    all_data = []
    token_lens = []
    max_token_len = 0
    max_span_len = 0

    for sentence in sentences:
        data = tokenizer({'token': sentence})
        all_data.append(data)

        token_len = len(data['token_input_ids'])
        token_lens.append(token_len)
        max_token_len = max(max_token_len, token_len)

        # Find maximum span length across all tokens
        for span in data['token_token_span']:
            max_span_len = max(max_span_len, len(span))

    # Now process with known max dimensions
    input_ids = []
    token_span = []

    for data in all_data:
        # Pad input ids
        padded_ids = data['token_input_ids'] + [0] * (max_token_len - len(data['token_input_ids']))
        input_ids.append(padded_ids)

        # Pad each span to max_span_len, then pad the whole array to max_token_len
        token_token_span = data['token_token_span']
        padded_spans = [span + [0] * (max_span_len - len(span)) for span in token_token_span]

        # Add padding spans if needed
        while len(padded_spans) < max_token_len:
            padded_spans.append([0] * max_span_len)

        token_span.append(padded_spans)



    # Convert to numpy arrays
    lens_np = np.array([len(data['token_token_span']) for data in all_data], dtype=np.int64)
    input_ids_np = np.array(input_ids, dtype=np.int64)
    token_span_np = np.array(token_span, dtype=np.int64)



    logits, _ = SESSION.run(None, { 'lens': lens_np, 'input_ids': input_ids_np, 'token_span': token_span_np })
    predictions = logits.argmax(-1)

    output = decode_output(logits.argmax(-1), _, all_data)
    print(output)
    return output

def parse_file(inp_path, progress=''):
    out_path = inp_path.replace('.ele_b.tok', '.msra.ner')

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
    outputs = [json.dump(x) for x in results]
    print('\n'.join(outputs))

    # with open(out_path, 'w') as f:
    #   f.write('\n'.join(outputs))

    return True


# file_paths = glob.glob('/data/hanlp/**/*.ele_b.tok')
# file_count = len(file_paths)

# for index, file_path in enumerate(file_paths):
#     try:
#         if parse_file(file_path, f'- <{index + 1}/{file_count}>') and (index + 1) % 100 == 0:
#             torch.cuda.empty_cache()
#             gc.collect()

#     except Exception as e:
#         print(f'Error processing {file_path}: {e}')
#         continue


parse_file('/data/hanlp/uaa/11302833-153.ele_b.tok', '1/1')
