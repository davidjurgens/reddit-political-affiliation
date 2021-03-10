import sys

sys.path.append('/home/zbohan/projects/')
#from src.models.textclassifier.create_train_test_dev_all_months import get_file_handle
import json
from sklearn.metrics import classification_report
from sklearn.utils import resample
from json import JSONDecodeError
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModel, AutoTokenizer,BertForSequenceClassification

test_mode=1

def downsampling(data):
    data_majority = data[data.Label == 1]
    data_minority = data[data.Label == 0]
    majority_down_sampled = resample(data_majority, n_samples=len(data_minority), random_state=42)
    return pd.concat([majority_down_sampled, data_minority])


def prepare_features(line,max_length=50):
    # input_ids = tokenizer.encode(line)
    # input_ids = input_ids[0:max_length - 1]
    # while (len(input_ids) < max_length - 1):
    #     input_ids.append(0)
    # input_ids.append(2)
    # return torch.tensor(input_ids)
    tokens=tokenizer.tokenize(line)[0:max_length-2]
    tokens=[tokenizer.cls_token]+tokens
    tokens.append(tokenizer.sep_token)

    while (len(tokens) < max_length):
        tokens.append(tokenizer.pad_token)
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
    return input_ids

class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        utterance = self.data.Text[index]
        y = self.data.Label[index]
        X = prepare_features(utterance)
        return X, y

    def __len__(self):
        return self.len

class BertTune(nn.Module):
    def __init__(self, BertModel,max_length=32):
        super(BertTune, self).__init__()
        self.BertModel=BertModel
        self.emb_dimension=768
        self.max_length=max_length
        self.lin1=nn.Linear(self.emb_dimension,2)
        self.relu=nn.Tanh()
        #self.dropout=nn.Dropout(0.5)
        self.maxpool=nn.MaxPool1d(self.max_length)
        self.init_emb()

    def init_emb(self):
        init_range = 0.5 / self.emb_dimension
        self.lin1.weight.data.uniform_(-init_range, init_range)

    def forward(self,input):
        bert_output=self.maxpool(self.BertModel(input)[0].permute(0,2,1)).squeeze(2)
        return self.lin1(self.relu(bert_output))

def evaluate(model, data):
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    model.eval()
    for sent, label in tqdm(data):
        if torch.cuda.is_available():
            sent = sent.cuda()
            label = label.cuda()
        label = label.long()
        output = model.forward(sent)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted.cpu() == label.cpu()).sum()
        y_pred.extend(list(predicted.cpu().numpy()))
        y_true.extend(list(label.cpu().numpy()))
    accuracy = 100.00 * correct.numpy() / total
    #print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
    print("Confusion Metrics \n", classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)['macro avg']['f1-score']



train_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/train.json'
test_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/test.json'
dev_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/dev.json'
comments_dir = '/shared/0/projects/reddit-political-affiliation/data/bert-text-classify/'


if __name__ == '__main__':
    train = json.load(open(train_dir))
    test = json.load(open(test_dir))
    dev = json.load(open(dev_dir))

    year_month='2019-05'
    dv="cuda:4"
    load_from=-1

    train_data = pd.read_csv(comments_dir + year_month + '/train.csv', index_col=0, sep='\t', engine='python')
    train_data = downsampling(train_data)
    print(train_data.Label.value_counts())
    test_data = pd.read_csv(comments_dir + year_month + '/test.csv', index_col=0, sep='\t')
    dev_data = pd.read_csv(comments_dir + year_month + '/dev.csv', index_col=0, sep='\t')
    print(train_data.shape, test_data.shape, dev_data.shape)

    device = torch.device(dv)
    torch.cuda.set_device(int(dv[-1]))

    BertTweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",normalization=True)
    model=BertTune(BertTweet)

    train_set = Intents(train_data)
    test_set = Intents(test_data)
    dev_set = Intents(dev_data)
    params = {'batch_size': 64, 'shuffle': True, 'drop_last': False, 'num_workers': 1}
    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)
    dev_loader = DataLoader(dev_set, **params)
    print(len(train_loader), len(test_loader), len(dev_loader))
    if not test_mode:
        if load_from != -1:
            model.load_state_dict(
                torch.load(comments_dir + year_month + "/" + str(load_from) + '.pt', map_location=device))
            print("load from" + str(load_from) + ".pt")
        model.cuda()

        loss_function = nn.CrossEntropyLoss()
        learning_rate = 1e-05
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
        iter_length = len(train_loader)
        max_epochs = 10
        try:
            best = 0
            for epoch in range(max_epochs):
                model.train()
                print("EPOCH -- {}".format(epoch))
                for i, (sent, label) in tqdm(enumerate(train_loader), total=iter_length):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        sent = sent.cuda()
                        label = label.cuda()
                    label = label.long()
                    output = model.forward(sent)
                    _, predicted = torch.max(output, 1)
                    loss = loss_function(output, label)
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % 100 == 0:
                        print(loss)

                print("Evaluation on dev set:")
                mc = evaluate(model, dev_loader)
                if mc > best:
                    print("Updating Best Score:", str(mc), "saving model...")
                    torch.save(model.state_dict(),
                               comments_dir + year_month + "/BertTweet_" + str(epoch + load_from + 1) + ".pt")
                    best = mc

        except KeyboardInterrupt:
            torch.save(model.state_dict(), comments_dir + year_month + "/BertTweet_" + "finished.pt")
            print("Evaluation on test set:")
            mc = evaluate(model, test_loader)
        print("Evaluation on test set:")
        mc = evaluate(model, test_loader)
    else:
        model.load_state_dict(torch.load(comments_dir+year_month+"/BertTweet_0.pt", map_location=device))
        model.cuda()
        print("Evaluation on test set:")
        mc = evaluate(model, test_loader)