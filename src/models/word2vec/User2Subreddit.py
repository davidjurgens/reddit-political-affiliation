import torch
import torch.nn as nn


class User2Subreddit(nn.Module):

    def __init__(self, num_users, emb_dimension, num_subreddits):
        super(User2Subreddit, self).__init__()
        self.num_users = num_users
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(num_users, emb_dimension)
        self.v_embeddings = nn.Embedding(num_subreddits, emb_dimension)
        self.political_layer = nn.Linear(emb_dimension, 1)
        self.init_emb()

    def init_emb(self):
        init_range = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-init_range, init_range)
        self.v_embeddings.weight.data.uniform_(-init_range, init_range)
        self.political_layer.weight.data.uniform_(-init_range, init_range)

    def forward(self, user_id, subreddit_id, political_user_ids):
        emb_u = self.u_embeddings(user_id)
        emb_v = self.v_embeddings(subreddit_id)

        # This this seems like the fastest way to do batch dot product:
        # https://github.com/pytorch/pytorch/issues/18027
        score = (emb_u * emb_v).sum(-1)
        score = torch.sigmoid(score)

        # If we have political users to predict for
        if political_user_ids.sum() > 0:
            emb_p = self.u_embeddings(political_user_ids)
            political_predictions = self.political_layer(emb_p)
            political_predictions = torch.sigmoid(political_predictions)
        else:
            political_predictions = None

        return score, political_predictions
