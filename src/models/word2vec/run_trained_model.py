import torch
from models.word2vec.User2Subreddit import User2Subreddit
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.word2vec.make_dataset import build_dataset

# TODO: Move settings to argparse
# So many paths....
year_month = ''
network_path = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + year_month + '_filtered.tsv'
flair_directory = '/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/' + year_month + '.tsv'
model_path = '/shared/0/projects/reddit-political-affiliation/working-dir/word2vec-outputs/' + year_month + '/9.pt'
id_mappings_path = '/shared/0/projects/reddit-political-affiliation/data/word2vec/dataset/' + year_month
embedding_dim = 50  # Constant
out_dir = '/shared/0/projects/reddit-political-affiliation/data/word2vec/predictions/users_' + year_month + '.tsv'

print("Building the dataset for {}".format(year_month))
dataset, training, validation, pol_validation, vocab = build_dataset(network_path, flair_directory)
dataset.load_id_mappings(id_mappings_path)

print("Loading in the model for {}".format(year_month))
model = User2Subreddit(dataset.num_users(), embedding_dim, dataset.num_users())
model.load_state_dict(torch.load(model_path))
model.eval()


def predict_on_political_validation():
    user_ids, pol_labels = [], []

    for user, pol_label in pol_validation.items():
        try:
            # User subreddit dataset spans 1 month. Political data spans the year. Some users might not be present
            user_ids.append(dataset.user_to_idx[user])
            pol_labels.append(pol_label)
        except KeyError:
            pass

    user_ids = torch.LongTensor(user_ids)
    pol_labels = torch.FloatTensor(pol_labels)

    emb_p = model.u_embeddings(user_ids)
    political_predictions = model.political_layer(emb_p)
    political_predictions = torch.sigmoid(political_predictions)

    preds = []
    for val in political_predictions.detach().numpy():
        if val[0] >= 0.5:
            preds.append(1)
        else:
            preds.append(0)

    labels = pol_labels.detach().numpy().astype(int)
    print("Accuracy for {} is {}".format(year_month, accuracy_score(labels, preds)))


def predict_user_affiliations(model, dataset):
    user_predictions = {}
    loader = DataLoader(dataset, batch_size=512)

    idx_to_user = {v: k for k, v in dataset.user_to_idx.items()}

    for i, data in enumerate(tqdm(loader, desc="Predicting user politics", total=len(dataset) / 512)):
        user_sub, politics_labels, subreddit_labels = data
        user_ids = user_sub[:, 0]
        subreddit_ids = user_sub[:, 1]

        _, pol_preds = model(user_ids, subreddit_ids, political_user_ids=user_ids)

        user_ids = user_ids.detach().numpy()

        for j in range(len(user_ids)):
            user = idx_to_user[user_ids[j]]
            user_predictions[user] = pol_preds[j][0]

    return user_predictions


def save_predictions_to_tsv(user_predictions, out_dir):
    with open(out_dir, 'w') as f:
        for user, score in user_predictions.items():
            f.write("{}\t{}\n".format(user, score))


if __name__ == '__main__':
    predict_on_political_validation()
    user_predictions = predict_user_affiliations(model, dataset)
    save_predictions_to_tsv(user_predictions, out_dir)
