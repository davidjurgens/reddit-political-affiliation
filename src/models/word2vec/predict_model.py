import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.word2vec.train_settings import device


def predict_user_affiliations(model, dataset, out_dir):
    user_predictions = {}
    loader = DataLoader(dataset, batch_size=100)

    idx_to_user = {v: k for k, v in dataset.user_to_idx.items()}

    for i, data in enumerate(tqdm(loader, desc="Predicting user politics", total=len(dataset) / 100)):
        user_sub, politics_labels, subreddit_labels = data

        user_ids = user_sub[:, 0].to(device)
        subreddit_ids = user_sub[:, 1].to(device)
        _, pol_preds = model(user_ids, subreddit_ids, political_user_ids=user_ids)

        user_ids = user_ids.cpu().detach().numpy()
        pol_preds = pol_preds.cpu().detach().numpy()

        for j in range(len(user_ids)):
            user = idx_to_user[user_ids[j]]
            user_predictions[user] = pol_preds[j]

    predictions_to_tsv(user_predictions, out_dir)


def output_top_n_similar(model, subreddit, all_subreddits, word_to_ix, n):
    cosine_sims = {}
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    print('looking up ', subreddit)
    
    try:
        sub_tensor = torch.tensor([word_to_ix[subreddit]], dtype=torch.long).to(device)
    except KeyError:
        return

    for sub in all_subreddits:
        ix = word_to_ix[sub]
        # print(ix)
        lookup_tensor = torch.tensor([ix,], dtype=torch.long).to(device)
        cos_result = cos(model.u_embeddings(sub_tensor), model.u_embeddings(lookup_tensor))
        cosine_sims[sub] = cos_result

    # Sort and grab the top N
    cosine_sims = {k: v for k, v in sorted(cosine_sims.items(), key=lambda item: item[1], reverse=True)}
    top_results = dict(itertools.islice(cosine_sims.items(), n))

    # Spit out the results to the console
    print("Top {} similar embeddings for the subreddit: {}".format(n, subreddit))
    for sub, score in top_results.items():
        print(sub, score)


def predictions_to_tsv(results, out_path):
    print("Writing predictions to TSV")
    with open(out_path + 'predictions.tsv', 'w') as f:
        for user, pred in results.items():
            f.write("%s\t%d\n" % (user, pred))
