import os

# os.environ["HANLP_HOME"] = "/2tb/var.hanlp/.hanlp"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hanlp, torch
tok_task = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE)

with open("sample.out") as file:
  sentences = [line.strip('\n') for line in file.readlines()]

# sentences = sentences[0:15]
import time
start = time.time()

result = tok_task(sentences)

end = time.time()
print(end - start)

output = ['|'.join(x) for x in result]
print('\n'.join(output))
