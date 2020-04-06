import random, json

def load_data_sst():
    # training data
    path = "train-nodes.tsv"
    data_train_all_nodes = []  
    with open(path) as file:
        for line in file.readlines()[1:]:
            data_train_all_nodes.append({
                "sentence": line.split("\t")[0],
                "label": int(line.split("\t")[1])
            })   
     
    # train/dev/test data
    for subset in ["train", "dev", "test"]:
        path = "{}.txt".format(subset)
        data = []  
        with open(path) as file:
            for line in file.readlines():
                segs = line[:-1].split(" ")
                tokens, word_labels = [], []
                label = int(segs[0][1])
                if label < 2: 
                    label = 0
                elif label >= 3: 
                    label = 1
                else: 
                    continue
                for i in range(len(segs) - 1):
                    if segs[i][0] == "(" and segs[i][1] in ["0", "1", "2", "3", "4"]\
                            and segs[i + 1][0] != "(":
                        tokens.append(segs[i + 1][:segs[i + 1].find(")")])
                        word_labels.append(int(segs[i][1]))
                data.append({
                    "label": label,
                    "sentence": " ".join(tokens),
                    "word_labels": word_labels
                })
        if subset == "train":
            data_train = data
        elif subset == "dev":
            data_dev = data
        else:
            data_test = data

    return data_train_all_nodes, data_train, data_dev, data_test

def read_scores(split):
    res = {}
    with open('{}_lm_scores.txt'.format(split)) as file:
        line = file.readline().strip().split('\t')
        while True:
            if len(line) < 2: break
            sentence = line[-1]
            tokens = sentence.lower().split(' ')
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
    return res

data_train_all_nodes, data_train, data_dev, data_test = load_data_sst()
candidates_dev = read_scores('dev')
candidates_test = read_scores('test')
for example in data_dev:
    example['candidates'] = candidates_dev[example['sentence']]
for example in data_test:
    example['candidates'] = candidates_test[example['sentence']]
with open('train_all_nodes.json', 'w') as file:
    file.write(json.dumps(data_train_all_nodes))
with open('train.json', 'w') as file:
    file.write(json.dumps(data_train))
with open('dev.json', 'w') as file:
    file.write(json.dumps(data_dev))
with open('test.json', 'w') as file:
    file.write(json.dumps(data_test))
