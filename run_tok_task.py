#!/usr/bin/python3

import os, gc, sys, glob
import hanlp, torch


TOK = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE)

def tokenize(inp_path, label):
  out_path = inp_path.replace('.txt', '.ele_b.tok')

  if os.path.isfile(out_path):
    print(f'{label} skipping [{out_path}]')
    return

  with open(inp_path, 'r', encoding='UTF-8') as inp_file:
    lines = inp_file.read().split('\n')

  out_data = TOK(lines)
  out_text = ''

  for out_line in out_data:
    out_text += '\t'.join(out_line)
    out_text += '\n'

  with open(out_path, 'w', encoding='UTF-8') as file:
    file.write(out_text)

  print(f'{label} [{out_path}] parsed and saved!')
  torch.cuda.empty_cache()
  gc.collect()

file_paths = glob.glob('/data/token/*/*.txt')
file_count = len(file_paths)

for index, file_path in enumerate(file_paths):
  tokenize(file_path, f'- <{index + 1}/{file_count}>')
