#!/usr/bin/python3

import os

# os.environ["HANLP_HOME"] = "/2tb/var.hanlp/.hanlp"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hanlp, torch
print(torch.cuda.is_available())

tok_task = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL)

with open("sample.out") as file:
  sentences = file.readlines()

import time
start = time.time()

result = tok_task(sentences)

end = time.time()
print(end - start)

# print(result)
