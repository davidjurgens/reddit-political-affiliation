import json
from collections import Counter
from tqdm import tqdm
import numpy as np

year_month = '2019-04'
network_path = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + year_month + '_filtered.tsv'
flair_directory = '/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/' + year_month + '.tsv'


def filter_data(all_count, sample):
    result = {}
    y = []
    for user in tqdm(sample):
        if user in all_count:
            result[user] = all_count[user]
        else:
            result[user] = Counter()
        y.append(sample[user])
    return result, y


def tokens2id(start, dim, sorted_count):
    uni2id = {}
    bi2id = {}
    count = 0
    id2token={}
    for idx, (k, v) in enumerate(sorted_count):
        if count == dim:
            break
        if idx >= start:
            if ' ' in k:
                bi2id[k] = count
            else:
                uni2id[k] = count
            id2token[count]=k
            count += 1
    return uni2id, bi2id,id2token


def convert_to_matrix(raw_data, uni2id, bi2id, dim):
    matrix = np.zeros((len(raw_data), dim))
    for idx, user in enumerate(tqdm(raw_data)):
        user_x = np.zeros(dim)
        raw_feat = raw_data[user]
        for w in raw_feat.keys():
            try:
                if ' ' in w:
                    if w in bi2id:
                        user_x[bi2id[w]] = raw_feat[w]
                else:
                    if w in uni2id:
                        user_x[uni2id[w]] = raw_feat[w]
            except KeyError:
                pass
        # user_x=np.concatenate((emb_p,user_x))
        matrix[idx] = user_x
    return matrix


def build_train_test_dev(user_words_dir, word_count_dir ,train, test, dev):
    all_count = json.load(open(user_words_dir))
    print(len(all_count))
    filter_train, train_y = filter_data(all_count, train)
    filter_test, test_y = filter_data(all_count, test)
    filter_dev, dev_y = filter_data(all_count, dev)
    # word_count = Counter()
    # for key in tqdm(filter_train):
    #     word_count.update(filter_train[key])
    # with open(
    #         '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/month_users_words/all_month_word_counts.json',
    #         'w') as fp:
    #     json.dump(word_count, fp)
    word_count=Counter(json.load(open(word_count_dir)))
    sorted_count = word_count.most_common()
    print(len(sorted_count))
    start, dim = 0,10000
    print("from", str(start),"to",str(start+dim))
    uni2id, bi2id,id2token = tokens2id(start, dim, sorted_count)
    train_matrix = convert_to_matrix(filter_train, uni2id, bi2id, dim)
    test_matrix = convert_to_matrix(filter_test, uni2id, bi2id, dim)
    dev_matrix = convert_to_matrix(filter_dev, uni2id, bi2id, dim)
    return (train_matrix,train_y), (test_matrix,test_y), (dev_matrix,dev_y),id2token