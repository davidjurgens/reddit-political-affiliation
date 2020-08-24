import argparse

import torch
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Train User2Subreddit word2vec model')
parser.add_argument('--network', type=str, help="Location of the bipartite network file between users and subreddits")
parser.add_argument('--flairs', type=str, help="Location of the user flair affiliations")
parser.add_argument('--out', type=str, help="Output directory")
parser.add_argument('--logdir', type=str, help="Log directory for tensorboard",
                    default='/home/kalkiek/projects/reddit-political-affiliation/tensorboard-logs/')
args = parser.parse_args()

embedding_dim = 50
batch_size = 4000
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
EPOCHS = 15
year = '2018'
month = '05'

# User / subreddit data file
if args.network:
    network_path = args.network
else:
    network_path = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + year + '-' + month + \
                   '_filtered.tsv'

# User politics data file
if args.flairs:
    flair_directory = args.flairs
else:
    flair_directory = "/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/" + year + "*.tsv"

if args.out:
    out_dir = args.out
else:
    out_dir = "/shared/0/projects/reddit-political-affiliation/data/word2vec/" + year + "-" + month + "/"

    # Tensorboard
    writer = SummaryWriter(logdir=args.logdir)

device = torch.device("cuda:5")
