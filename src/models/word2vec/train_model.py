import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split

from src.data.make_dataset import build_dataset
from src.models.word2vec.User2Subreddit import User2Subreddit
from src.models.word2vec.train_settings import *

training_dataset, training, validation, vocab = build_dataset(network_path, flair_directory)

train_loader = DataLoader(training, **params)
validation_loader = DataLoader(validation, **params)
iter_length = len(training) / batch_size

# Model parameters set in train_settings
model = User2Subreddit(training_dataset.num_users(), embedding_dim, training_dataset.num_subreddits())
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
loss_function = nn.BCELoss()


def training_iteration(epoch, i, model, data, loss, pol_loss):
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
    subreddit_preds, pol_preds, subreddit_labels = model(user_ids, subreddit_ids, political_ids, subreddit_labels)

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


def validation_iteration(epoch, i, model, sample_size):
    # Select a random sample from the validation data
    sample, _ = random_split(validation, [sample_size, len(validation) - sample_size])

    model.train(False)
    _, val_loss, val_pol_loss = training_iteration(epoch, i, model, data=sample, loss=0, pol_loss=0)

    writer.add_scalar('validation word2vec loss', val_loss.cpu().detach().numpy().item(),
                      i * batch_size + epoch * len(training))
    if val_pol_loss:
        writer.add_scalar('validation political loss', val_pol_loss.cpu().detach().numpy().item(),
                          i * batch_size + epoch * len(training))


if __name__ == '__main__':

    for epoch in tqdm(range(EPOCHS), desc='Epoch'):
        loss, p_loss = 0, 0
        for i, data in enumerate(tqdm(train_loader, total=iter_length)):
            model, loss, p_loss = training_iteration(epoch, i, model, data, loss, p_loss)

            writer.add_scalar('word2vec loss', loss.cpu().detach().numpy().item(),
                              i * batch_size + epoch * len(training))

            if i % 1000 == 0:
                print(' loss at step %d: %f' % (i, loss.cpu().detach().numpy()))

            # Every 10 percent output the validation loss of a random sample
            if (i / iter_length) % 0.1 == 0:
                validation_iteration(epoch, i, model, sample_size=10000)

        # Save the model after every epoch
        torch.save(model.state_dict(), out_dir + str(epoch) + ".pt")

    writer.close()
