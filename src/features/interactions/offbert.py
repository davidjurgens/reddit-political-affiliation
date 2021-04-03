import sys
sys.path.append('/home/zbohan/projects/')
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer,BertForSequenceClassification, BertConfig, AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report
#from transformers import AutoModel, AutoTokenizer, AdamW

class BertTune(nn.Module):
    def __init__(self, BertModel,max_length=32):
        super(BertTune, self).__init__()
        self.BertModel=BertModel
        self.emb_dimension=768
        self.max_length=max_length
        self.lin1=nn.Linear(self.emb_dimension,2)
        self.relu=nn.Tanh()
        self.dropout=nn.Dropout(0.1)
        self.maxpool=nn.MaxPool1d(self.max_length)
        self.init_emb()

    def init_emb(self):
        init_range = 0.5 / self.emb_dimension
        self.lin1.weight.data.uniform_(-init_range, init_range)

    def forward(self,input):
        bert_output=self.maxpool(self.BertModel(input)[0].permute(0,2,1)).squeeze(2)
        return self.lin1(self.dropout(self.relu(bert_output)))


def prepare_features(seq_1, max_seq_length=128, zero_pad=True, include_CLS_token=True, include_SEP_token=True):
    # Tokenize Input
    tokens = tokenizer.tokenize(seq_1)[0:max_seq_length - 2]
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

    def __getitem__(self, index):
        utterance = self.data.tweet[index]
        y =1 if self.data.subtask_a[index]=='OFF' else 0
        X, mask = prepare_features(utterance)
        return X, mask, y

    def __len__(self):
        return self.len

def load_train(train_dir):
    train_df = pd.read_csv(train_dir, sep='\t')
    return Intents(train_df)

def load_test(test_text_dir, test_label_dir):
    test_text = pd.read_csv(test_text_dir, sep='\t')
    test_label= pd.read_csv(test_label_dir, sep=',')
    text_df=test_text.join(test_label.set_index('id'), on='id')
    #print(text_df)
    return Intents(text_df)

def evaluate(model, data):
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    model.eval()
    phar=0.8
    for sent,mask,label in tqdm(data):
        if torch.cuda.is_available():
            sent = sent.cuda()
            label = label.cuda()
            mask=mask.cuda()
        label = label.long()
        output = model.forward(sent,attention_mask=mask)[0]

        soft_output=F.softmax(output,dim=1)
        weighted=torch.tensor([phar,1-phar]).cuda().repeat(soft_output.shape[0]).view(soft_output.shape[0],-1)
        weighted_output=soft_output*weighted
        _, predicted = torch.max(weighted_output, 1)

        total += label.size(0)
        correct += (predicted.cpu() == label.cpu()).sum()
        y_pred.extend(list(predicted.cpu().numpy()))
        y_true.extend(list(label.cpu().numpy()))
    accuracy = 100.00 * correct.numpy() / total
    # print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.item(), accuracy))
    print("Confusion Metrics \n", classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)['macro avg']['f1-score']

if __name__ == '__main__':
    test_mode=0
    dv = "cuda:7"

    train_dir='OLIDv1.0/olid-training-v1.0.tsv'
    test_text_dir='OLIDv1.0/testset-levela.tsv'
    test_label_dir='OLIDv1.0/labels-levela.csv'


    train_set=load_train(train_dir)
    test_set=load_test(test_text_dir,test_label_dir)
    train_set,dev_set=random_split(train_set,[len(train_set)-len(test_set),len(test_set)],generator=torch.Generator().manual_seed(42))

    device = torch.device(dv)
    torch.cuda.set_device(int(dv[-1]))

    config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification(config)

    # BertTweet = AutoModel.from_pretrained("vinai/bertweet-base")
    # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    # model = BertTune(BertTweet)
    model.cuda()

    params = {'batch_size': 16, 'shuffle': True, 'drop_last': False, 'num_workers': 1}
    train_loader = DataLoader(train_set, **params)
    dev_loader=DataLoader(dev_set,**params)
    test_loader=DataLoader(test_set,**params)
    print(len(train_loader),len(dev_loader),len(test_loader))

    loss_function = nn.CrossEntropyLoss()
    learning_rate = 1e-5
    optimizer = AdamW(params=model.parameters(), lr=learning_rate)
    iter_length = len(train_loader)
    max_epochs = 10

    if test_mode:
        best = 0
        for epoch in range(max_epochs):
            model.train()
            print("EPOCH -- {}".format(epoch))
            for i, (sent, mask, label) in tqdm(enumerate(train_loader), total=iter_length):
                optimizer.zero_grad()
                #print(sent.shape,mask.shape,label)
                #sent = sent.squeeze(1)
                if torch.cuda.is_available():
                    sent = sent.cuda()
                    label = label.cuda()
                    mask=mask.cuda()
                label = label.long()
                output = model.forward(sent,attention_mask=mask)[0]
                _, predicted = torch.max(output, 1)
                loss = loss_function(output, label)
                loss.backward()
                optimizer.step()


            print("Evaluation on dev set:")
            mc = evaluate(model, dev_loader)
            if mc > best:
                print("Updating Best Score:", str(mc), "saving model...")
                torch.save(model.state_dict(),"best_bert.pt")
                best = mc
    else:
        model.load_state_dict(torch.load("best_bert.pt", map_location=device))
        print("Evaluation on test set:")
        mc = evaluate(model, test_loader)

