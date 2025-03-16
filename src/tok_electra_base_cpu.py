import numpy as np
import onnxruntime as ort
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer

onnx_path = "./model/tok-ctb9-electra-base.onnx"
session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

names = [input.name for input in session.get_inputs()]
shapes = [input.shape for input in session.get_inputs()]
types = [input.type for input in session.get_inputs()]

tokenizer = TransformerSequenceTokenizer("./tokenizer/electra", 'token', use_fast=True, ret_subtokens=True)

def export_outputs(predictions, sentences, batch_offsets, lens):
    chunks = []
    start = 0

    for i, length in enumerate(lens):
        end = start + length
        prediction = predictions[start+1:end-1]
        start = end

        sentence = sentences[i]
        sent_offsets = batch_offsets[i]

        chunk = []
        acc = ''

        for j, t in enumerate(prediction):
            offset = sent_offsets[j]
            raw = sentence[offset[0]:offset[1]]

            if acc and (t == 0 or t == 2):  # B or S
                chunk.append(acc)
                acc = ''

            acc += raw

        if acc:
            chunk.append(acc)

        chunks.append(chunk)

    return chunks

def process_batch(sentences):
    input_ids = []
    batch_offsets = []

    for sentence in sentences:
        data = tokenizer({'token': sentence})
        input_ids.extend(data['token_input_ids'])
        batch_offsets.append(data['token_subtoken_offsets'])

    # max_len = max(len(ids) for ids in input_ids)
    # input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]

    lens = [len(x) + 2 for x in batch_offsets]
    lens = np.array(lens, dtype=np.int64)

    input_ids = np.array(input_ids, dtype=np.int64)
    logits, _ = session.run(None, {'lens': lens, 'input_ids': input_ids})

    predictions = logits[0].argmax(-1).tolist()
    return export_outputs(predictions, sentences, batch_offsets, lens)

with open("sample.out") as file:
    sentences = [line.strip('\n') for line in file.readlines()]

import time
start = time.time()

batch_size = 1
results = []

for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]
    batch_results = process_batch(batch)
    results.extend(['\t'.join(x) for x in batch_results])

end = time.time()
print(end - start)

print('\n'.join(results))
