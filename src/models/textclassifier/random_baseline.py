import sys

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# cong_dir = '/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/'
# train_cong = pd.read_csv(cong_dir + 'train.tsv', sep='\t')
# dev_cong = pd.read_csv(cong_dir + 'dev.tsv', sep='\t')
# test_cong = pd.read_csv(cong_dir + 'test.tsv', sep='\t')
#
# train_user = set(train_cong['username'])
# dev_user = set(dev_cong['username'])
# test_user = set(test_cong['username'])
#
# sorted_train_cong=train_cong.sort_values(["username","source"])
# distinct_train_cong=sorted_train_cong.drop_duplicates(subset="username",keep="first").sample(frac=1)
# sorted_test_cong = test_cong.sort_values(["username", "source"])
# distinct_test_cong = sorted_test_cong.drop_duplicates(subset="username", keep="first").sample(frac=1)
# sorted_dev_cong = dev_cong.sort_values(["username", "source"])
# distinct_dev_cong = sorted_dev_cong.drop_duplicates(subset="username", keep="first").sample(frac=1)
#
# print(distinct_train_cong.shape,distinct_test_cong.shape,distinct_dev_cong.shape)

saved_path = '/shared/0/projects/reddit-political-affiliation/data/bert-text-classify/users_prediction/'
evaluating = 'flair'  # can be flair gold silver all
eval_from_flair = pd.read_csv(saved_path + "Roberta_flair_downsampling_" + evaluating + ".tsv", sep='\t')
eval_from_flair = eval_from_flair.drop_duplicates(subset="username", keep="last").reset_index()
print(eval_from_flair)
label = eval_from_flair['politics']
# rand=np.random.randint(2, size=len(label))
rand = 1 - np.zeros(len(label))
crf = classification_report(label, rand, output_dict=True)
crf2 = classification_report(label, rand, output_dict=False)
print(crf)
print(crf2)
