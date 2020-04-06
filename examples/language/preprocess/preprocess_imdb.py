import json, os
from tqdm import tqdm
from nltk import word_tokenize

def read_candidates(split):
    res, count = {}, {}
    cc = 0
    with open('./lm-{}/scores.txt'.format(split)) as file:
        line = file.readline().strip().split('\t')
        while True:
            if len(line) < 2: break
            sentence = line[-1]
            tokens = sentence.split(' ')
            candidates = [[] for i in range(len(tokens))]
            while True:
                line = file.readline().strip().split('\t')
                if len(line) != 4: break
                pos, word, score = int(line[1]), line[2], float(line[3])
                if score == float('-inf'):
                    continue
                if len(candidates[pos]) == 0:
                    if word != tokens[pos]:
                        continue
                elif score < candidates[pos][0][1] - 5.0:
                    continue
                candidates[pos].append((word, score))
            res[sentence] = [[w[0] for w in cand] for cand in candidates]
            if not sentence in count:
                count[sentence] = 1
            else:
                count[sentence] += 1
            cc += 1
    return res, count

def read_data(split, candidates, fix_text):
    if split == 'test':
        subdir = 'test'
    else:
        subdir = 'train'
    with open(os.path.join('aclImdb', subdir, 'imdb_%s_files.txt' % split)) as f:
      filenames = [line.strip() for line in f]
    data = []
    num_words = 0
    for fn in tqdm(filenames):
      label = 1 if fn.startswith('pos') else 0
      with open(os.path.join('aclImdb', subdir, fn)) as f:
        x_raw = f.readlines()[0].strip().replace('<br />', ' ')
        x_toks = word_tokenize(x_raw)
        num_words += len(x_toks)
        sentence = ' '.join(x_toks)
        if sentence in candidates:
            cand = candidates[sentence]
        elif "".join(sentence.split()) in fix_text and \
                fix_text["".join(sentence.split())] in candidates:
            sentence = fix_text["".join(sentence.split())]
            cand = candidates[sentence]
        else:
            cand = None
        data.append({
            'sentence': sentence,
            'label': label,
            'candidates': cand
        })
    # num_pos = sum(y for x, y in data)
    # num_neg = sum(1 - y for x, y in data)
    # avg_words = num_words / len(data)
    # print('Read %d examples (+%d, -%d), average length %d words' % (
    #     len(data), num_pos, num_neg, avg_words))
    return data        

for split in ['train', 'dev', 'test']:
    print('Processing split {}'.format(split))
    candidates, count = read_candidates(split)
    fix_text = {}
    for key in candidates:
        fix_text["".join(key.split())] = key
    data = read_data(split, candidates, fix_text)
    cnt = 0
    for example in data:
        if example['candidates'] is not None:
            if count[example['sentence']] == 0:
                example['candidates'] = None
            else:
                count[example['sentence']] -= 1
                cnt += 1
    print('{} examples with candidates'.format(cnt))
    with open('{}.json'.format(split), 'w') as file:
        file.write(json.dumps(data))
