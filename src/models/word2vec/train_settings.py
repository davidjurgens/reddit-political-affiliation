import argparse

import os

import torch
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Train User2Subreddit word2vec model')
parser.add_argument('--network', type=str, help="Location of the bipartite network file between users and subreddits")
parser.add_argument('--flairs', type=str, help="Location of the user flair affiliations")
parser.add_argument('--num_epochs', type=int, help="The number of epochs to run", default=10)
parser.add_argument('--batch_size', type=int, help="The batch size", default=512)
parser.add_argument('--out', type=str, help="Output directory")
parser.add_argument('--device', type=str, help="The GPU to run on (e.g., cuda:0)")
parser.add_argument('--year_month', type=str, help="The year-month (YYYY-MM) of Reddit data to analyze",
                    default="2018-10")
parser.add_argument('--max_users', type=int, help="The maximum number of users to train on", default=-1)
parser.add_argument('--log_dir', type=str, help="Log directory for tensorboard",
                    default='/shared/0/projects/reddit-political-affiliation/working-dir/tensorboard-logs/')
parser.add_argument('--data_directory', type=str,
                    default='/shared/0/projects/reddit-political-affiliation/data/word2vec/dataset/')
args = parser.parse_args()

embedding_dim = 50
batch_size = args.batch_size
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 2}
EPOCHS = args.num_epochs
year_month = args.year_month
data_directory = args.data_directory + year_month + '/'

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

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

log_dir = args.log_dir + '/' + year_month
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Tensorboard
writer = SummaryWriter(logdir=log_dir)

if args.device:
    torch.cuda.set_device(int(args.device[-1]))
    device = torch.device(args.device)
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
