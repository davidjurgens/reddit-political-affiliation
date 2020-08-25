import argparse

import torch
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Train User2Subreddit word2vec model')
parser.add_argument('--network', type=str, help="Location of the bipartite network file between users and subreddits")
parser.add_argument('--flairs', type=str, help="Location of the user flair affiliations")
parser.add_argument('--out', type=str, help="Output directory")
parser.add_argument('--device', type=str, help="The GPU to run on (e.g., cuda:0)")
parser.add_argument('--year_month', type=str, help="The year-month (YYYY-MM) of Reddit data to analyze", default="2018-05")
parser.add_argument('--month', type=str, help="The month of Reddit data to analyze", default="5")
parser.add_argument('--logdir', type=str, help="Log directory for tensorboard",
                    default='/home/kalkiek/projects/reddit-political-affiliation/tensorboard-logs/')
args = parser.parse_args()

embedding_dim = 50
batch_size = 4000
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
EPOCHS = 5
year_month = args.year_month

# User / subreddit data file
if args.network:
    network_path = args.network
else:
    network_path = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + year_month + \
                   '_filtered.tsv'

# User politics data file
if args.flairs:
    flair_directory = args.flairs
else:
    flair_directory = "/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/" + year_month + ".tsv"

if args.out:
    out_dir = args.out
else:
    out_dir = "/shared/0/projects/reddit-political-affiliation/data/word2vec/" + year_month + "/"

    # Tensorboard
    writer = SummaryWriter(logdir=args.logdir)

if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda:0")
