# 每月的ground——pol load进来
# split them all
# 在所有month上dump data然后save 大userwords

import io
import sys

sys.path.append('/home/zbohan/projects/')
from src.data.make_dataset import read_flair_political_affiliations, dict_random_split
from src.models.textclassifier.feature import build_train_test_dev
import glob
import json
from json import JSONDecodeError
import bz2
import lzma
import zstandard as zstd
from collections import Counter, defaultdict
from nltk import word_tokenize
from nltk.util import ngrams
import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

preparing_data = 0


def save_data(path, data_dict):
    with open(path, 'w') as fp:
        json.dump(data_dict, fp)


def get_file_handle(file_path):
    ext = file_path.split('.')[-1]

    if ext == "bz2":
        return bz2.open(file_path)
    elif ext == "xz":
        return lzma.open(file_path)
    elif ext == "zst":
        f = open(file_path, 'rb')
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        return text_stream

    raise AssertionError("Invalid extension for " + file_path + ". Expecting bz2 or xz file")


def get_word_frequencies(file_pointer, ground_pol):
    user_word = defaultdict(Counter)
    left_word_freq, right_word_freq = Counter(), Counter()
    for count, line in enumerate(file_pointer):
        try:
            submission = json.loads(f.readline().strip())
            username, text = submission['author'], submission['body']

            if username in ground_pol:
                unigram = word_tokenize(text)
                bigram = ngrams(unigram, 2)
                for uni in unigram:
                    if len(uni) <= 20 and len(uni) > 1 and 'http' not in uni:  # remove some noisy term and hyper link
                        user_word[username][uni] += 1
                        if ground_pol[username] == 1:
                            right_word_freq[uni] += 1
                        else:
                            left_word_freq[uni] += 1
                for bi in bigram:
                    if len(bi[0]) <= 20 and len(bi[1]) <= 20 and 'http' not in bi[0] and 'http' not in bi[1]:
                        user_word[username][bi[0] + ' ' + bi[1]] += 1


        except (JSONDecodeError, AttributeError) as e:
            print("Failed to parse line: {} with error: {}".format(line, e))

        if count % 1000000 == 0 and count > 0:
            print("Completed %d lines" % (count))
    return user_word, left_word_freq, right_word_freq


if __name__ == '__main__':
    train_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/train.json'
    test_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/test.json'
    dev_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/dev.json'
    all_count_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/month_users_words/all_month_user.json'
    word_count_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/month_users_words/all_month_word_counts.json'

    if preparing_data:
        z = {}
        all_count = {}
        for i in range(1, 13):
            # print("********************")
            year_month = '2019-' + ('0' if i < 10 else '') + str(i)
            network_path = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + year_month + '_filtered.tsv'
            if i < 7:
                flair_directory = '/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/' + year_month + '.tsv'
            else:
                flair_directory = '/home/zbohan/projects/src/data/' + year_month + '.tsv'
            flair_files = glob.glob(flair_directory)
            user_words_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg-feat/' + year_month + '.json'
            ground_pol = read_flair_political_affiliations(flair_files)
            z = {**z, **ground_pol}
            file_path = '/shared/2/datasets/reddit-dump-all/RC/RC_' + year_month + ('.xz' if i < 7 else '.zst')
            print(file_path)
            f = get_file_handle(file_path)
            this_month_user_word, _, _ = get_word_frequencies(f, ground_pol)
            for user in this_month_user_word:
                all_count[user] = this_month_user_word[user] + (all_count[user] if user in all_count else Counter())
            print(len(all_count))
        save_data(all_count_dir, all_count)

        print(len(z))
        train_z, left_z = dict_random_split(z, 0.8)
        test_z, dev_z = dict_random_split(left_z, 0.5)
        print(len(train_z), len(test_z), len(dev_z))

        save_data(train_dir, train_z)
        save_data(test_dir, test_z)
        save_data(dev_dir, dev_z)
    else:
        train = json.load(open(train_dir))
        test = json.load(open(test_dir))
        dev = json.load(open(dev_dir))
        count_0 = 0
        for user in train:
            if train[user] == 0:
                count_0 += 1
        print(count_0 / len(train), 1 - count_0 / len(train))
        print(len(train), len(test), len(dev))

        (train_matrix,train_y),(test_matrix,test_y),(dev_matrix,dev_y),id2token=build_train_test_dev(all_count_dir,word_count_dir,train,test,dev)
        print(train_matrix.shape,test_matrix.shape,dev_matrix.shape)

        print("Training...patience...")
        clf = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight={0:1,1:6.4},n_jobs=8).fit(train_matrix, train_y)
        print("Done, let's see!!")

        pre_y=clf.predict(test_matrix)
        print("Confusion Metrics \n", classification_report(test_y,pre_y))

        coef=clf.coef_
        #print(coef)
        print("sorting coef...")
        arg_sort=np.argsort(-np.absolute(coef[0]))
        high_weight_word=[]
        hww=100
        for i in range(hww):
            high_weight_word.append(id2token[arg_sort[i]])
        print(hww,"highest weighted word:")
        print(high_weight_word)


        with open('/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/coef.pkl', 'wb') as file:
            pickle.dump(clf, file)