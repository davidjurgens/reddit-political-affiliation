import sys

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')
from src.features.interactions.political_comment import PoliticalComment
from sklearn.metrics import classification_report
from sklearn.utils import resample
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import numpy as np


def downsampling(data):
    data_majority = data[
        data.politics_y == ("Republican" if training_name == "flair" or training_name == "community" else "Democrat")]
    data_minority = data[
        data.politics_y == ("Democrat" if training_name == "flair" or training_name == "community" else "Republican")]
    majority_down_sampled = resample(data_majority, n_samples=len(data_minority), random_state=42)
    return pd.concat([majority_down_sampled, data_minority], ignore_index=True).sample(frac=1)


def read_in_comments(in_file, count=-1):
    comments = []
    with open(in_file, 'r', encoding="utf-8") as f:
        for line in tqdm(f, total=count if count > 0 else 137629803):
            line = line.strip()
            try:
                comment_id, parent_id, username, subreddit, created, politics, text = line.split('\t')
                political_comment = PoliticalComment(comment_id, parent_id, username, subreddit, created, politics,
                                                     text)
                comments.append(political_comment.to_dict())
                if count > 0 and len(comments) >= count:
                    print("Total number of political comments: {}".format(len(comments)))
                    return comments
            except Exception:
                pass

    print("Total number of political comments: {}".format(len(comments)))
    return comments


def prepare_features(seq_1, max_seq_length=64, zero_pad=True, include_CLS_token=True, include_SEP_token=True):
    # Tokenize Input
    # print(seq_1)
    tokens = tokenizer.tokenize(str(seq_1))[0:max_seq_length - 2]
    # Initialize Tokens
    if include_CLS_token:
        tokens.insert(0, tokenizer.cls_token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    # Input Mask
    input_mask = [1] * len(input_ids)
    # Zero-pad sequence length
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_id)
            input_mask.append(0)
    # print(torch.tensor(input_ids).shape)
    return torch.tensor(input_ids), torch.tensor(input_mask)


class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        self.Label_Map = {'Democrat': 0, 'Republican': 1}

    def __getitem__(self, index):
        utterance = self.data.text[index]
        y = self.Label_Map[self.data.politics_y[index]]
        X, mask = prepare_features(utterance)
        user_name = self.data.username[index]
        return X, mask, y, user_name

    def __len__(self):
        return self.len


# class BertTune(nn.Module):
#     def __init__(self, BertModel,max_length=64):
#         super(BertTune, self).__init__()
#         self.BertModel=BertModel
#         self.emb_dimension=768
#         self.max_length=max_length
#         self.lin1=nn.Linear(self.emb_dimension,2)
#         self.relu=nn.Tanh()
#         #self.dropout=nn.Dropout(0.5)
#         self.maxpool=nn.MaxPool1d(self.max_length)
#         self.init_emb()
#
#     def init_emb(self):
#         init_range = 0.5 / self.emb_dimension
#         self.lin1.weight.data.uniform_(-init_range, init_range)
#
#     def forward(self,input,attention_mask=None):
#         bert_output=self.maxpool(self.BertModel(input,attention_mask=attention_mask)[0].permute(0,2,1)).squeeze(2)
#         return [self.lin1(self.relu(bert_output))]

def evaluate_by_user(model, data, name=""):
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    pred_label = []
    user_name_list = []
    # text_list=[]
    model.eval()
    # with torch.no_grad():
    for sent, mask, label, user_name in tqdm(data):
        # print(sent.shape)
        # origin_sent=tokenizer.batch_decode(sent)
        # print(origin_sent)
        sent = sent.cuda()
        label = label.cuda()
        mask = mask.cuda()
        label = label.long()

        output = model.forward(sent, attention_mask=mask)[0]

        soft_output = F.softmax(output, dim=1)
        party_score, predicted = torch.max(soft_output, 1)
        y_pred.extend(party_score.detach().cpu().numpy())
        pred_label.extend(list(predicted.detach().cpu().numpy()))

        y_true.extend(list(label.detach().cpu().numpy()))
        user_name_list.extend(user_name)
        # text_list.extend(origin_sent)

        del sent
        del mask
        del label

    # print(user_name_list,y_pred,y_true,pred_label)
    result_pd = pd.DataFrame(
        {'username': user_name_list, 'party_score': y_pred, 'predict_politics': pred_label, 'politics': y_true})
    saved_path = "/shared/0/projects/reddit-political-affiliation/data/bert-text-classify/users_prediction/"
    if len(name) > 1:
        result_pd.to_csv(saved_path + model_name + "_" + name + "_" + str(load_from) + ".tsv", sep="\t")
    result_pd = result_pd.sort_values(by=["username", "party_score"])
    # print(result_pd)
    result_pd = result_pd.drop_duplicates(subset="username", keep="last").reset_index()
    if len(name) > 1:
        result_pd.to_csv(saved_path + model_name + "_most_confident_" + name + "_" + str(load_from) + ".tsv", sep="\t")
    # print (result_pd)

    user_y_pred = list(result_pd['predict_politics'])
    user_y_true = list(result_pd['politics'])
    print("Confusion Metrics \n", classification_report(user_y_true, user_y_pred))
    mc = classification_report(user_y_true, user_y_pred, output_dict=True)
    # print(mc)
    return mc  # ['macro avg']['f1-score']


def evaluate(model, data):
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    # with torch.no_grad():
    model.eval()
    for sent, mask, label, user_name in tqdm(data):
        sent = sent.cuda()
        label = label.cuda()
        mask = mask.cuda()
        label = label.long()

        output = model.forward(sent, attention_mask=mask)[0]
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted.detach().cpu() == label.detach().cpu()).sum()
        y_pred.extend(list(predicted.detach().cpu().numpy()))
        y_true.extend(list(label.detach().cpu().numpy()))

        del sent
        del mask
        del label

    accuracy = 100.00 * correct.numpy() / total
    # print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
    print("Confusion Metrics \n", classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)['macro avg']['f1-score']


comments_dir = '/shared/0/projects/reddit-political-affiliation/data/bert-text-classify/'

# gold/flair-9
if __name__ == '__main__':
    dv = "cuda:4"
    load_from = 9
    test_mode = 1

    model_name = ""

    remove_self_declare = True
    if remove_self_declare:
        model_name += "Remove_self_declare"

    in_file = '/shared/0/projects/reddit-political-affiliation/data/interactions/20_comments' + (
        "_drop_self_declare" if remove_self_declare else "") + '.tsv'
    is_20_comments = True
    if is_20_comments:
        model_name += "20Comments_"
        df_comments = pd.read_csv(in_file, sep="\t")
        print("Finish Data Loading", df_comments.shape)
    else:
        model_name += "allComments"
        comments = read_in_comments(in_file, count=-1)
        df_comments = pd.DataFrame(comments)

    print(df_comments)

    cong_dir = '/shared/0/projects/reddit-political-affiliation/data/conglomerate-affiliations/'
    train_cong = pd.read_csv(cong_dir + 'train.tsv', sep='\t')
    dev_cong = pd.read_csv(cong_dir + 'dev.tsv', sep='\t')
    test_cong = pd.read_csv(cong_dir + 'test.tsv', sep='\t')

    train_user = set(train_cong['username'])
    dev_user = set(dev_cong['username'])
    test_user = set(test_cong['username'])

    train_comments = df_comments[df_comments['username'].isin(train_user)]
    dev_comments = df_comments[df_comments['username'].isin(dev_user)]
    test_comments = df_comments[df_comments['username'].isin(test_user)]

    print(train_comments.shape, dev_comments.shape, test_comments.shape)

    community_train_cong = train_cong[train_cong['source'] == 'community']
    community_dev_cong = dev_cong[dev_cong['source'] == 'community']
    community_test_cong = test_cong[test_cong['source'] == 'community']

    community_merged_train = pd.merge(train_comments, community_train_cong, on='username')[
        ['username', 'text', 'politics_y', 'source', 'subreddit_x']]
    community_merged_test = pd.merge(test_comments, community_test_cong, on='username')[
        ['username', 'text', 'politics_y', 'source', 'subreddit_x']]
    community_merged_dev = pd.merge(dev_comments, community_dev_cong, on='username')[
        ['username', 'text', 'politics_y', 'source', 'subreddit_x']]

    train_cong = train_cong[train_cong['source'] != 'community']
    dev_cong = dev_cong[dev_cong['source'] != 'community']
    test_cong = test_cong[test_cong['source'] != 'community']

    sorted_train_cong = train_cong.sort_values(["username", "source"])
    distinct_train_cong = sorted_train_cong.drop_duplicates(subset="username", keep="first").sample(frac=1)
    sorted_test_cong = test_cong.sort_values(["username", "source"])
    distinct_test_cong = sorted_test_cong.drop_duplicates(subset="username", keep="first").sample(frac=1)
    sorted_dev_cong = dev_cong.sort_values(["username", "source"])
    distinct_dev_cong = sorted_dev_cong.drop_duplicates(subset="username", keep="first").sample(frac=1)

    merged_train = pd.merge(train_comments, distinct_train_cong, on='username')[
        ['username', 'text', 'politics_y', 'source', 'subreddit_x']]
    merged_test = pd.merge(test_comments, distinct_test_cong, on='username')[
        ['username', 'text', 'politics_y', 'source', 'subreddit_x']]
    merged_dev = pd.merge(dev_comments, distinct_dev_cong, on='username')[
        ['username', 'text', 'politics_y', 'source', 'subreddit_x']]

    merged_train = pd.concat([merged_train, community_merged_train])
    merged_test = pd.concat([merged_test, community_merged_test])
    merged_dev = pd.concat([merged_dev, community_merged_dev])

    # print(merged_train.head())

    device = torch.device(dv)
    torch.cuda.set_device(int(dv[-1]))

    # model_name="Tweet_Bert"
    # BertTweet = AutoModel.from_pretrained("vinai/bertweet-base")
    # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",normalization=True)
    # model=BertTune(BertTweet)
    # only_politics = False
    # if only_politics:
    #     merged_train = merged_train[merged_train['subreddit_x'] == 'politics']
    #     #print (merged_train)
    #     #merged_test = merged_train[merged_test['subreddit'] == 'politics']
    #     #merged_dev = merged_train[merged_dev['subreddit'] == 'politics']
    #     model_name += "_politics"

    model_name += "Roberta_tune"
    # config = RobertaConfig.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    training_name = 'community'
    model_name += "_" + training_name
    training_data = merged_train[merged_train['source'] == training_name].reset_index()
    if training_name == 'silver':
        additional_training_data = merged_train[merged_train['source'] == 'gold'].reset_index()
        model_name += '_full'
        training_data = pd.concat([training_data, additional_training_data])
    print(training_data)

    is_downsampling = True
    if is_downsampling:
        training_data = downsampling(training_data)
        model_name += "_downsampling"

    print(training_data['politics_y'].value_counts())

    print("Full Model Name:", model_name)
    # text_set=set(merged_train['text'])
    # total_len_list=[]
    # for text in text_set:
    #     words=word_tokenize(text)
    #     total_len_list.append(len(words))
    # print(np.mean(total_len_list),np.std(total_len_list),np.percentile(total_len_list,80))

    print(training_data)
    train_set = Intents(training_data.sample(frac=1))
    test_set = Intents(merged_test)
    dev_set = Intents(merged_dev)
    params = {'batch_size': 64, 'shuffle': True, 'drop_last': False, 'num_workers': 0}
    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)
    dev_loader = DataLoader(dev_set, **params)
    print(len(train_loader), len(test_loader), len(dev_loader))

    if not test_mode:
        if load_from != -1:
            model.load_state_dict(
                torch.load(comments_dir + model_name + "_Text_Classifier_" + str(load_from) + '.pt',
                           map_location=device))
            print("load from" + str(load_from) + ".pt")
        model.cuda()

        loss_function = nn.CrossEntropyLoss()
        learning_rate = 1e-05
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
        iter_length = len(train_loader)
        max_epochs = 10

        try:
            best = 0
            for epoch in range(max_epochs - load_from - 1):
                model.train()
                print("EPOCH -- {}".format(epoch))
                total_loss = 0
                for i, (sent, mask, label, _) in tqdm(enumerate(train_loader), total=iter_length):
                    optimizer.zero_grad()
                    sent = sent.cuda()
                    mask = mask.cuda()
                    label = label.cuda()
                    label = label.long()
                    output = model.forward(sent, attention_mask=mask)[0]
                    _, predicted = torch.max(output, 1)
                    loss = loss_function(output, label)
                    total_loss += loss.data.item()
                    loss.backward()
                    optimizer.step()

                    # del sent
                    # del mask
                    # del label

                    # if (i + 1) % 100 == 0:
                    #     print(loss)

                print("Evaluation on dev set:(pass)")
                # mc = evaluate(model, dev_loader)
                # if mc > best:
                #     print("Updating Best Score:", str(mc), "saving model...")
                torch.save(model.state_dict(),
                           comments_dir + model_name + "_Text_Classifier_" + str(epoch + load_from + 1) + ".pt")
                #     best = mc
                print("Loss in this epoch:", total_loss / iter_length)
        except KeyboardInterrupt:
            torch.save(model.state_dict(), comments_dir + model_name + "_Text_Classifier_" + "finished.pt")
            # print("Evaluation on test set:")
            # mc = evaluate(model, test_loader)
        # print("Evaluation on test set:")
        # mc = evaluate(model, test_loader)
    else:
        model.load_state_dict(
            torch.load(comments_dir + model_name + "_Text_Classifier_" + str(load_from) + ".pt", map_location=device))
        model.cuda()
        # for name,param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name,param.data)
        # print (model.parameters())
        print("Evaluation on test set:")

        gold_test = merged_test[merged_test['source'] == 'gold'].reset_index()
        silver_test = merged_test[merged_test['source'] == 'silver'].reset_index()
        flair_test = merged_test[merged_test['source'] == 'flair'].reset_index()
        community_test = merged_test[merged_test['source'] == 'community'].reset_index()

        print(gold_test)

        gold_set = Intents(gold_test)
        silver_set = Intents(silver_test)
        flair_set = Intents(flair_test)
        community_set = Intents(community_test)

        gold_loader = DataLoader(gold_set, **params)
        silver_loader = DataLoader(silver_set, **params)
        flair_loader = DataLoader(flair_set, **params)
        community_loader = DataLoader(community_set, **params)

        print("Evaluating on gold...")
        mc = evaluate_by_user(model, gold_loader, "gold")
        print("Evaluating on silver...")
        mc = evaluate_by_user(model, silver_loader, "silver")
        print("Evaluating on flair...")
        mc = evaluate_by_user(model, flair_loader, "flair")
        print("Evaluating on community...")
        mc = evaluate_by_user(model, community_loader, "community")
        # print("Evaluating on all test...")
        # mc = evaluate(model, test_loader)
