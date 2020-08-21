import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.make_dataset import build_dataset
from src.models.word2vec.User2Subreddit import User2Subreddit
from src.models.word2vec.train_settings import *

training_dataset, training, validation, vocab = build_dataset(network_directory, flair_directory, months=1)

train_loader = DataLoader(training, **params)
validation_loader = DataLoader(validation, **params)
iter_length = len(training) / batch_size

# Model parameters set in train_settings
model = User2Subreddit(training_dataset.num_users(), embedding_dim, training_dataset.num_subreddits())
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
word_to_ix = {word: i for i, word in enumerate(vocab)}
loss_function = nn.BCELoss()


for epoch in tqdm(range(EPOCHS), desc='Epoch'):

    total_loss, pol_loss = 0, 0

    for i, data in enumerate(tqdm(train_loader, total=iter_length)):
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

        writer.add_scalar('word2vec loss', loss.cpu().detach().numpy().item(),
                          i * batch_size + epoch * len(training))

        if i % 1000 == 0:
            print(' loss at step %d: %f' % (i, loss.cpu().detach().numpy()))

    # Save the model after every epoch
    torch.save(model.state_dict(), out_dir + str(epoch) + ".pt")

writer.close()
