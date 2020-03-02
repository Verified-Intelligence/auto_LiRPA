import random
from auto_LiRPA.utils import logger

def load_data_sst():
    # training data
    path = "data/sst/train-nodes.tsv"
    logger.info("Loading data {}".format(path))
    data_train_warmup = []  
    with open(path) as file:
        for line in file.readlines()[1:]:
            data_train_warmup.append({
                "sentence": line.split("\t")[0],
                "label": int(line.split("\t")[1])
            })   
     
    # train/dev/test data
    for subset in ["train", "dev", "test"]:
        path = "data/sst/{}.txt".format(subset)
        logger.info("Loading data {}".format(path))
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
        for example in data:
            for i, token in enumerate(example["sentence"]):
                if token == "-LRB-":
                    example["sentence"][i] = "("
                if token == "-RRB-":
                    example["sentence"][i] = ")"
        if subset == "train":
            data_train = data
        elif subset == "dev":
            data_dev = data
        else:
            data_test = data

    return data_train_warmup, data_train, data_dev, data_test

def load_data(dataset):
    if dataset == "sst":
        return load_data_sst()

def get_batches(data, batch_size):
    batches = []
    random.shuffle(data)
    for i in range((len(data) + batch_size - 1) // batch_size):
        batches.append(data[i * batch_size : (i + 1) * batch_size])
    return batches