import onnxruntime as ort
import numpy as np
import time
import json
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

SESSION = ort.InferenceSession("./model/pos-ctb9-electra-small.onnx", OPTIONS, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) #

tokenizer = TransformerSequenceTokenizer("./tokenizer/electra", 'token', ret_token_span=True, ret_subtokens=False, ret_mask_and_type=False, ret_subtokens_group=False, ret_prefix_mask=False,)

with open("./config/pos/vocabs-ctb.json", "r") as f:
    VOCABS = json.load(f)["tag"]["idx_to_token"]

def process_sentences(sentences):
    # First, collect all data and determine max dimensions
    all_data = []
    max_token_len = 0
    max_span_len = 0

    for sentence in sentences:
        data = tokenizer({'token': sentence})

        all_data.append(data)
        max_token_len = max(max_token_len, len(data['token_input_ids']))

        # Find maximum span length across all tokens
        for span in data['token_token_span']:
            max_span_len = max(max_span_len, len(span))

    # Now process with known max dimensions
    input_ids = []
    token_spans = []

    for data in all_data:
        # Pad input ids
        padded_ids = data['token_input_ids'] + [0] * (max_token_len - len(data['token_input_ids']))
        input_ids.append(padded_ids)

        # Pad each span to max_span_len, then pad the whole array to max_token_len
        token_token_span = data['token_token_span']
        padded_spans = []

        for span in token_token_span:
            padded_span = span + [0] * (max_span_len - len(span))
            padded_spans.append(padded_span)

        # Add padding spans if needed
        while len(padded_spans) < max_token_len:
            padded_spans.append([0] * max_span_len)

        token_spans.append(padded_spans)

    # Convert to numpy arrays
    lens_np = np.array([len(data['token_token_span']) for data in all_data], dtype=np.int64)
    input_ids_np = np.array(input_ids, dtype=np.int64)
    token_span_np = np.array(token_spans, dtype=np.int64)

    # logits, _ = SESSION.run(None, {
    #     'lens': lens_np,
    #     'input_ids': input_ids_np,
    #     'token_span': token_span_np
    # })

    # Convert NumPy arrays to CUDA tensors and back
    lens_cuda = torch.tensor(lens_np, device='cuda').cpu().numpy()
    input_ids_cuda = torch.tensor(input_ids_np, device='cuda').cpu().numpy()
    token_span_cuda = torch.tensor(token_span_np, device='cuda').cpu().numpy()

    logits, _ = SESSION.run(None, {
        'lens': lens_cuda,
        'input_ids': input_ids_cuda,
        'token_span': token_span_cuda
    })
    predictions = logits.argmax(-1)

    results = []
    for i, pred in enumerate(predictions):
        results.append([VOCABS[idx] for idx in pred[0:lens_np[i]]])

    return results

inp_file = '/mnt/ssd00/Works/hanlp/toks/uaa/11302833-65.ele_b.tok'
with open(inp_file, 'r') as f:
    lines = f.readlines()
    sentences = [line.strip().split('\t') for line in lines]

import time
start = time.time()

results = process_sentences(sentences)
print(time.time() - start)

# outputs = ['\t'.join(x) for x in results]
# print('\n'.join(outputs))
