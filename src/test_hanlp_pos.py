import os

# os.environ["HANLP_HOME"] = "/2tb/var.hanlp/.hanlp"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hanlp, torch
pos_task = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

inp_file = '/mnt/ssd00/Works/hanlp/toks/uaa/11302833-65.ele_b.tok'
with open(inp_file, 'r') as f:
    lines = f.readlines()
    sentences = [line.strip().split('\t') for line in lines]

# sentences = sentences[0:15]
import time
start = time.time()

result = pos_task(sentences)

end = time.time()
print(end - start)

output = ['\t'.join(x) for x in result]
print('\n'.join(output))
