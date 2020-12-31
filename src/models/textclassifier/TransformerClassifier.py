import sys

sys.path.append('/home/zbohan/projects/')
from src.models.textclassifier.create_train_test_dev_all_months import get_file_handle
import json
from sklearn.metrics import classification_report
from json import JSONDecodeError
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import RobertaTokenizer
from pytorch_transformers import RobertaForSequenceClassification, RobertaConfig
import pandas as pd

train_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/train.json'
test_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/test.json'
dev_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/log-reg/save_all_users/dev.json'
comments_dir = '/shared/0/projects/reddit-political-affiliation/data/user-comments/'
preparing = 0
test_mode = 1


def get_comments(file_pointer, ground_pol):
    text_list = []
    y = []
    for count, line in enumerate(file_pointer):
        try:
            submission = json.loads(f.readline().strip())
            username, text = submission['author'], submission['body']
            if username in ground_pol:
                text_list.append(text)
                y.append(ground_pol[username])
        except (JSONDecodeError, AttributeError) as e:
            print("Failed to parse line: {} with error: {}".format(line, e))

        if count % 1000000 == 0 and count > 0:
            print("Completed %d lines" % (count))
    d = {'Text': text_list, 'Label': y}
    return pd.DataFrame(d)


def prepare_features(seq_1, max_seq_length=50,
                     zero_pad=True, include_CLS_token=True, include_SEP_token=True):
    ## Tokenzine Input
    tokens = tokenizer.tokenize(seq_1)[0:max_seq_length - 2]
    ## Initialize Tokens
    if include_CLS_token:
        tokens.insert(0, tokenizer.cls_token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    ## Input Mask
    input_mask = [1] * len(input_ids)
    ## Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    # print(torch.tensor(input_ids).shape)
    return torch.tensor(input_ids), input_mask


class Intents(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        utterance = self.data.Text[index]
        y = self.data.Label[index]
        X, _ = prepare_features(utterance)
        return X, y

    def __len__(self):
        return self.len


def evaluate(model, data):
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    model.eval()
    for sent, label in tqdm(data):
        sent = sent.squeeze(1)
        if torch.cuda.is_available():
            sent = sent.cuda()
            label = label.cuda()
        label = label.long()
        output = model.forward(sent)[0]
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted.cpu() == label.cpu()).sum()
        y_pred.extend(list(predicted.cpu().numpy()))
        y_true.extend(list(label.cpu().numpy()))
    accuracy = 100.00 * correct.numpy() / total
    # print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
    print("Confusion Metrics \n", classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)['macro avg']['f1-score']


if __name__ == '__main__':
    train = json.load(open(train_dir))
    test = json.load(open(test_dir))
    dev = json.load(open(dev_dir))

    year_month = '2019-05'
    dv = "cuda:7"
    load_from = 14

    year_month = '2019-05'
    dv = "cuda:1"
    load_from = 14
    if preparing:
        file_path = '/shared/2/datasets/reddit-dump-all/RC/RC_' + year_month + (
            '.xz' if year_month[-1] < '7' else '.zst')
        print(file_path)
        f = get_file_handle(file_path)
        train_data = get_comments(f, train)
        f = get_file_handle(file_path)
        test_data = get_comments(f, test)
        f = get_file_handle(file_path)
        dev_data = get_comments(f, dev)
        train_data.to_csv(comments_dir + year_month + '/train.csv', sep='\t')
        test_data.to_csv(comments_dir + year_month + '/test.csv', sep='\t')
        dev_data.to_csv(comments_dir + year_month + '/dev.csv', sep='\t')

    else:
        train_data = pd.read_csv(comments_dir + year_month + '/train.csv', index_col=0, sep='\t', engine='python')
        test_data = pd.read_csv(comments_dir + year_month + '/test.csv', index_col=0, sep='\t')
        dev_data = pd.read_csv(comments_dir + year_month + '/dev.csv', index_col=0, sep='\t')
        print(train_data.shape, test_data.shape, dev_data.shape)

        device = torch.device(dv)
        torch.cuda.set_device(int(dv[-1]))
        config = RobertaConfig.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification(config)
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

            max_epochs = 50

            try:
                best = 0
                for epoch in range(max_epochs):
                    model = model.train()
                    print("EPOCH -- {}".format(epoch))
                    for i, (sent, label) in tqdm(enumerate(train_loader), total=iter_length):
                        optimizer.zero_grad()
                        sent = sent.squeeze(1)
                        if torch.cuda.is_available():
                            sent = sent.cuda()
                            label = label.cuda()
                        label = label.long()
                        output = model.forward(sent)[0]
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
                                   comments_dir + year_month + "/" + str(epoch + load_from + 1) + ".pt")
                        best = mc

            except KeyboardInterrupt:
                torch.save(model.state_dict(), comments_dir + year_month + "/" + "finished.pt")
                print("Evaluation on test set:")
                mc = evaluate(model, test_loader)
            print("Evaluation on test set:")
            mc = evaluate(model, test_loader)
        else:
            model.load_state_dict(torch.load(comments_dir + year_month + "/17.pt", map_location=device))
            model.cuda()
            model.load_state_dict(torch.load(comments_dir + year_month + "/17.pt", map_location=device))

            print("Evaluation on test set:")
            mc = evaluate(model, test_loader)
