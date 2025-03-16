import json
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer


class TransformerNamedEntityRecognizer():
    def __init__(self):
        with open(vocabs_path, "r") as f:
            vocabs = json.load(f)
        self.idx_to_token = vocabs['tag']['idx_to_token']

    def decode_output(self, logits, mask, batch):
        if isinstance(logits, torch.Tensor):
            logits = logits.tolist()

        predictions = self.prediction_to_human(logits, batch)
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

    def prediction_to_human(self, logits, batch):
        predictions = []
        for sent_logits in logits:
            tags = [self.idx_to_token[idx] for idx in sent_logits]
            predictions.append(tags)
        return predictions

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

vocabs_path = "./config/ner/vocabs.json"
tokenizer_path = "./tokenizer/electra"
onnx_path = "./model/ner-msra-electra-small.onnx"

so = ort.SessionOptions()
so.intra_op_num_threads=1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider']) # CUDAExecutionProvider
transform = TransformerSequenceTokenizer(tokenizer_path, 'token', ret_subtokens=False, ret_subtokens_group=False, ret_token_span=True, use_fast=False)

ner = TransformerNamedEntityRecognizer()

def process_sentence(sentence):
    inputs = transform({'token': sentence})

    token_token_span = inputs['token_token_span']
    max_len = max(len(span) for span in token_token_span)
    padded_spans = [span + [0] * (max_len - len(span)) for span in token_token_span]

    lens = np.array([len(inputs['token_token_span'])], dtype=np.int64)

    input_ids = inputs['token_input_ids']
    input_ids = np.array([input_ids], dtype=np.int64)
    token_span = np.array([padded_spans], dtype=np.int64)

    logits, mask = session.run(None, { 'lens': lens, 'input_ids': input_ids, 'token_span': token_span })

    return ner.decode_output(logits.argmax(-1), mask, inputs)

sentence = ['而且', '还', '是', '每', '个', '月', '两万', '?', '!']
results = process_sentence(sentence)
print(results)
