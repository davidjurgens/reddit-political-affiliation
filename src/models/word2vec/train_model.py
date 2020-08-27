import random
import sys

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm

sys.path.append('../..')  # make it work in the root directory
sys.path.append('.')

from src.data.make_dataset import build_dataset
from src.models.word2vec.User2Subreddit import User2Subreddit
from src.models.word2vec.predict_model import predict_user_affiliations, top_n_similar_embeddings, \
    save_similar_embeddings_to_tsv
from src.models.word2vec.train_settings import *

from sklearn.metrics import auc, roc_curve

torch.manual_seed(42)

dataset, training, validation, pol_validation, vocab = build_dataset(network_path, flair_directory,
                                                                     max_users=args.max_users)
word_to_ix = {word: i for i, word in enumerate(vocab)}
all_subreddits = {v for v in vocab if v[:2] == 'r/' and v[2:4] != 'u_'}
print("# of subreddits: " + str(len(all_subreddits)))

# Model parameters set in train_settings
train_loader = DataLoader(training, **params)
validation_loader = DataLoader(validation, **params)
iter_length = len(training) / batch_size

model = User2Subreddit(dataset.num_users(), embedding_dim, dataset.num_subreddits())
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
loss_function = nn.BCELoss()


def training_iteration(epoch, i, model, data, pol_loss):
    optimizer.zero_grad()

    user_sub, politics_labels, subreddit_labels = data

    user_ids = user_sub[:, 0]
    subreddit_ids = user_sub[:, 1].to(device)
    subreddit_labels = subreddit_labels.type(torch.FloatTensor).to(device)

    # Grab the user IDs for those that had political labels
    p_indices = [i for i, v in enumerate(politics_labels) if v >= 0]
    political_ids = user_ids.index_select(0, torch.LongTensor(p_indices))
    user_ids = user_ids.to(device)
    political_ids = political_ids.to(device)
    subreddit_preds, pol_preds = model(user_ids, subreddit_ids, political_ids)

    loss = loss_function(subreddit_preds, subreddit_labels)

    # If we had some political users in this batch...
    if len(p_indices) > 0:
        pol_labels = torch.LongTensor([v for v in politics_labels if v >= 0]).float().to(device)

        # Squeeze call necessary to go from (k, 1) to (k) dimensions due to batching
        pol_loss = loss_function(pol_preds.squeeze(), pol_labels)

        writer.add_scalar('political loss', pol_loss.cpu().detach().numpy().item(),
                          i * batch_size + epoch * len(training))
        loss += pol_loss

    loss.backward()
    optimizer.step()

    return model, loss, pol_loss


def validation_iteration(epoch, model, sample_size):
    # Select a random sample from the validation data
    sample, _ = random_split(validation, [sample_size, len(validation) - sample_size])

    model.train(False)
    for i, sample_data in enumerate(DataLoader(sample, batch_size=sample_size)):
        _, val_loss, val_pol_loss = training_iteration(epoch, i, model, sample_data, pol_loss=0)

        writer.add_scalar('validation word2vec loss', val_loss.cpu().detach().numpy().item(),
                          i * batch_size + epoch * len(training))
        if val_pol_loss:
            writer.add_scalar('validation political loss', val_pol_loss.cpu().detach().numpy().item(),
                              i * batch_size + epoch * len(training))


def pol_validation_iteration(model, sample_size, step):
    # Switch to evaluation model
    model.eval()

    # Select a random sample from the political validation data
    validation_list = list(pol_validation.items())
    sample = dict(random.sample(validation_list, sample_size))

    # Build a dataset of the users in the validation set
    user_ids, pol_labels = [], []

    for user, pol_label in sample.items():
        try:
            # User subreddit dataset spans 1 month. Political data spans the year. Some users might not be present
            user_ids.append(dataset.user_to_idx[user])
            pol_labels.append(pol_label)
        except KeyError:
            pass

    user_ids = torch.LongTensor(user_ids).to(device)
    pol_labels = torch.FloatTensor(pol_labels).to(device)

    emb_p = model.u_embeddings(user_ids)
    political_predictions = model.political_layer(emb_p)
    political_predictions = torch.sigmoid(political_predictions)

    fpr, tpr, thresholds = roc_curve(pol_labels.cpu().detach().numpy(),
                                     political_predictions.cpu().detach().numpy(), pos_label=1)
    pol_auc = auc(fpr, tpr)

    # Predict the political affiliations and compute the loss
    pol_loss = loss_function(political_predictions.squeeze(), pol_labels).cpu().detach().numpy().item()
    print("Validation political loss at step %d: %f; AUC: %f" % (step, pol_loss, pol_auc))
    writer.add_scalar('validation political loss', pol_loss, step)
    writer.add_scalar('validation political AUC', pol_auc, step)

    # After evaluation, turn training back on
    model.train(True)


if __name__ == '__main__':

    sample_subreddits = ['r/nba', 'r/CryptoCurrency', 'r/Conservative', 'r/Liberal', 'r/AskReddit',
                         'r/Aww', 'r/Games', 'r/Hunting', 'r/Feminism', 'r/The_Donald',
                         'r/lawnmowers', 'r/juul', 'r/teenagers']

    for epoch in tqdm(range(EPOCHS), desc='Epoch'):
        p_loss = 0
        for i, data in enumerate(tqdm(train_loader, total=iter_length), 1):
            step = i * batch_size + epoch * len(training)
            model, loss, p_loss = training_iteration(epoch, i, model, data, p_loss)

            writer.add_scalar('word2vec loss', loss.cpu().detach().numpy().item(),
                              i * batch_size + epoch * len(training))

            if i % 100 == 0:
                print(' loss at epoch %d, step %d: %f; political loss: %f' \
                      % (epoch, i, loss.cpu().detach().numpy(),
                         p_loss.cpu().detach().numpy()))

            # if i % int(iter_length/10) == 0:
            if i % 200 == 0:
                pol_validation_iteration(model, sample_size=100, step=step)
                validation_iteration(epoch, model, sample_size=10000)
                if epoch > 3 and i % 10000 == 0:
                    [top_n_similar_embeddings(model, sub, all_subreddits, word_to_ix, n=10) for sub in
                     sample_subreddits]

        # Run all the validation stuff at the end of the epoch
        pol_validation_iteration(model, sample_size=100, step=step)
        validation_iteration(epoch, model, sample_size=10000)
        if epoch > 3:
            for sub in sample_subreddits:
                similar_subs = top_n_similar_embeddings(model, sub, all_subreddits, word_to_ix, n=10)
                save_similar_embeddings_to_tsv(sub, similar_subs, epoch, step, out_dir)
            [top_n_similar_embeddings(model, sub, all_subreddits, word_to_ix, n=10) for sub in sample_subreddits]

        # Save the model after every epoch
        torch.save(model.state_dict(), out_dir + str(epoch) + ".pt")

    writer.close()

    # When the model is done training predict all user affiliations
    predict_user_affiliations(model, dataset, out_dir=out_dir)
