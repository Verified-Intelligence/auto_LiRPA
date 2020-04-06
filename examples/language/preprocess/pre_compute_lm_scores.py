# Ref: https://worksheets.codalab.org/rest/bundles/0x3f614472f4a14393b3d85d5568114591/contents/blob/precompute_lm_scores.py

"""Precompute language model scores."""
import argparse
import json
import os
import sys
import torch
from tqdm import tqdm

from data_utils import load_data

sys.path.insert(0, 'tmp/windweller-l2w/adaptive_softmax')
import query as lmquery

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('--data', type=str, default='sst')
  parser.add_argument('--out', default='tmp')
  parser.add_argument('--window-radius', '-w', type=int, default=6)
  parser.add_argument('--neighbor-file', type=str, default='tmp/synonyms.json')
  return parser.parse_args()

def main():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  query_handler = lmquery.load_model(device)
  with open(OPTS.neighbor_file) as f:
    neighbors = json.load(f)

  data_train_warmup, data_train, data_dev, data_test = load_data(OPTS.data)
  split = [('train', data_train), ('dev', data_dev), ('test', data_test)]

  for s in split:
    data = s[1]
    out_file = os.path.join(OPTS.out, '{}_lm_scores.txt'.format(s[0]))

    with open(out_file, 'w') as f:
      for sent_idx, example in enumerate(tqdm(data)):
        sentence = example["sentence"]
        print('%d\t%s' % (sent_idx, sentence), file=f)
        words = sentence.lower().strip().split(' ')
        for i, w in enumerate(words):
          if w in neighbors:
            options = [w] + neighbors[w]
            start = max(0, i - OPTS.window_radius)
            end = min(len(words), i + 1 + OPTS.window_radius)
            # Remove OOV words from prefix and suffix
            prefix = [x for x in words[start:i] if x in query_handler.word_to_idx]
            suffix = [x for x in words[i+1:end] if x in query_handler.word_to_idx]
            queries = []
            in_vocab_options = []
            for opt in options:
              if opt in query_handler.word_to_idx:
                queries.append(prefix + [opt] + suffix)
                in_vocab_options.append(opt)
              else:
                print('%d\t%d\t%s\t%s' % (sent_idx, i, opt, float('-inf')), file=f)
            if queries:
              log_probs = query_handler.query(queries, batch_size=16)
              for x, lp in zip(in_vocab_options, log_probs):
                print('%d\t%d\t%s\t%s' % (sent_idx, i, x, lp), file=f)
        f.flush()

if __name__ == '__main__':
  OPTS = parse_args()
  main()