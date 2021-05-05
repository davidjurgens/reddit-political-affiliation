import sys

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')
from sklearn.metrics import classification_report
import pandas as pd

# from transformers import AutoModel, AutoTokenizer,BertForSequenceClassification
# from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
# from nltk.tokenize import word_tokenize


if __name__ == '__main__':
    saved_path = '/shared/0/projects/reddit-political-affiliation/data/bert-text-classify/users_prediction/'
    evaluating = 'silver'  # can be flair gold silver all
    eval_from_flair = pd.read_csv(saved_path + "Roberta_flair_downsampling_" + evaluating + ".tsv", sep='\t')
    eval_from_flair = eval_from_flair.drop_duplicates(subset="username", keep="last").reset_index()
    eval_from_gold = pd.read_csv(saved_path + "Roberta_gold_downsampling_" + evaluating + ".tsv", sep='\t')
    eval_from_gold = eval_from_gold.drop_duplicates(subset="username", keep="last").reset_index()
    # eval_from_silver = pd.read_csv(saved_path + "Roberta_silver_downsampling_" + evaluating + ".tsv", sep='\t')
    # eval_from_silver = eval_from_silver.drop_duplicates(subset="username", keep="last").reset_index()
    # print(eval_from_silver.shape)
    eval_all = pd.merge(eval_from_flair, eval_from_gold, on='username').drop(
        columns=['index_x', 'Unnamed: 0_x', 'index_y', 'Unnamed: 0_y'])
    # eval_all=pd.merge(eval_all,eval_from_silver,on='username').drop(columns=['index','Unnamed: 0'])
    eval_all.columns = ['username', 'predict_score_flair', 'predict_politics_flair', 'real_politics_flair',
                        'predict_score_gold', 'predict_politics_gold',
                        'real_politics']  # ,'predict_score_silver','predict_politics_silver','real_politics']
    eval_all.drop(columns=['real_politics_flair'], inplace=True)
    print(eval_all)
    saved_path = 'user_prediction/evaluating_on_' + evaluating + ".tsv"
    eval_all.to_csv(saved_path, sep='\t')
    print("Flair testing on " + evaluating)
    crf = classification_report(eval_all['real_politics'], eval_all['predict_politics_flair'], output_dict=True)
    crf2 = classification_report(eval_all['real_politics'], eval_all['predict_politics_flair'], output_dict=False)
    print(crf)
    print(crf2)
    print("Gold testing on " + evaluating)
    crf = classification_report(eval_all['real_politics'], eval_all['predict_politics_gold'], output_dict=True)
    crf2 = classification_report(eval_all['real_politics'], eval_all['predict_politics_gold'], output_dict=False)
    print(crf)
    print(crf2)

    # cong_dir = '/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/'
    # test_cong = pd.read_csv(cong_dir + 'test.tsv', sep='\t')
    # sorted_test_cong = test_cong.sort_values(["username", "source"])
    # distinct_test_cong = sorted_test_cong.drop_duplicates(subset="username", keep="first").sample(frac=1)
    # print(distinct_test_cong.shape)
