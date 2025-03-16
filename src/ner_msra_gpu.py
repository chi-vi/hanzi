import json
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer


with open("./config/ner/vocabs.json", "r") as f:
    VOCABS = json.load(f)["tag"]["idx_to_token"]

class TransformerNamedEntityRecognizer():
    def __init__(self):
        with open("./config/ner/vocabs.json", "r") as f:
            vocabs = json.load(f)
        self.idx_to_token = vocabs['tag']['idx_to_token']

    def decode_output(self, logits, mask, batch):
        if isinstance(logits, torch.Tensor):
            logits = logits.tolist()

        predictions = []
        for sent_logits in logits:
            tags = [self.idx_to_token[idx] for idx in sent_logits]
            predictions.append(tags)

        return self.tag_to_span(predictions, batch)

    def get_entities(self, tags):
        entities = []
        label = None
        start = None
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if label:
                    entities.append((label, start, i))
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

    def tag_to_span(self, batch_tags, batch):
        spans = []
        sents = batch['token']
        for tags, tokens in zip(batch_tags, sents):
            entities = self.get_entities(tags)
            spans.append(entities)
        return spans

    def decorate_spans(self, spans, batch):
        batch_ner = []
        for spans_per_sent, tokens in zip(spans, batch['token']):
            ner_per_sent = []
            for label, start, end in spans_per_sent:
                entity = ''.join(tokens[start:end])
                ner_per_sent.append((entity, label, start, end))
            batch_ner.append(ner_per_sent)
        return batch_ner


session = ort.InferenceSession("./model/ner-msra-electra-small.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

tokenizer = TransformerSequenceTokenizer("./tokenizer/electra", 'token', ret_subtokens=False, ret_subtokens_group=False, ret_token_span=True, use_fast=False)
# transform = TransformerSequenceTokenizer(tokenizer_path, 'token', ret_token_span=True, ret_subtokens=False, ret_mask_and_type=False, ret_subtokens_group=False, ret_prefix_mask=False,)

ner = TransformerNamedEntityRecognizer()

def process_sentences(sentences):
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

    input_ids = []
    token_span = []

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

        token_span.append(padded_spans)

    # Convert to numpy arrays
    lens_np = np.array([len(data['token_token_span']) for data in all_data], dtype=np.int64)
    input_ids_np = np.array(input_ids, dtype=np.int64)
    token_span_np = np.array(token_span, dtype=np.int64)

    logits, _ = SESSION.run(None, { 'lens': lens_np, 'input_ids': input_ids_np, 'token_span': token_span_np })

    results = []
    for i, pred in enumerate(predictions):
        results.append(pred[0:lens_np[i]])

    # return ner.decode_output(logits.argmax(-1), mask, inputs)

sentence = ["阿婆主", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司", "。"]
results = process_sentences([sentence])
print(results)
