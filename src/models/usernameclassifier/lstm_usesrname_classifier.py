import sys
from collections import defaultdict

import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from tqdm import tqdm

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.models.usernameclassifier.UsernameClassifier import UsernameClassifier

# SETTINGS
BASE_DIR = '/shared/0/projects/reddit-political-affiliation/data/username-labels/'

print("Building datasets for train, dev, and test")
TEXT = Field(tokenize=list, batch_first=True, include_lengths=True)
LABEL = LabelField(dtype=torch.float, batch_first=True)
fields = [('text', TEXT), ('label', LABEL), (None, None), ]

training_data = TabularDataset(BASE_DIR + 'user2label.silver.train.csv', format='csv', fields=fields,
                               skip_header=True)
dev_data = TabularDataset(BASE_DIR + 'user2label.silver.dev.csv', format='csv', fields=fields, skip_header=True)
test_data = TabularDataset(BASE_DIR + 'user2label.silver.test.csv', format='csv', fields=fields, skip_header=True)

print("Sample of preprocessed text: {}".format(vars(training_data.examples[0])))

print("Initializing glove embeddings")
TEXT.build_vocab(training_data, min_freq=1)
LABEL.build_vocab(training_data)

# No. of unique tokens in text
print("Size of TEXT vocabulary:", len(TEXT.vocab))

# No. of unique tokens in label
print("Size of LABEL vocabulary:", len(LABEL.vocab))

# Commonly used words
print("Commonly used words")
print(TEXT.vocab.freqs.most_common(10))

# Word dictionary
print(TEXT.vocab.stoi)

# check whether cuda is available
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# set batch size
BATCH_SIZE = 128

# Load an iterator
train_iterator, valid_iterator = BucketIterator.splits(
    (training_data, dev_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)

size_of_vocab = len(TEXT.vocab)
embedding_dim = 15
num_hidden_nodes = 256
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

# instantiate the model
model = UsernameClassifier(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers,
                           bidirectional=True, dropout=dropout)

# define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()


# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# push to cuda if available
model = model.to(device)
criterion = criterion.to(device)


def train(model, iterator, optimizer, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for batch in tqdm(iterator):
        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        loss = criterion(predictions, batch.label)

        # compute the binary accuracy
        acc = binary_accuracy(predictions, batch.label)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    # train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    # evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

# Check out some of the predictions
model.eval()

dev_pred_df = defaultdict(list)

# deactivates autograd
with torch.no_grad():
    for batch in tqdm(valid_iterator):

        # retrieve text and no. of words
        text, text_lengths = batch.text

        # convert to 1d tensor
        predictions = model(text, text_lengths).squeeze()

        for i, cids in enumerate(text.cpu()):
            username = ''.join([TEXT.vocab.itos[cid] for cid in cids])
            pred = predictions[i].cpu().item()
            dev_pred_df['username'].append(username)
            dev_pred_df['prediction'].append(pred)

        # print(text)
        # break
dev_pred_df = pd.DataFrame(dev_pred_df)
dev_pred_df.head()
