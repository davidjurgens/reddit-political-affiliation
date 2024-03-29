import argparse
import os

import torch
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Train User2Subreddit word2vec model')
parser.add_argument('--network', type=str, help="Location of the bipartite network file between users and subreddits")
parser.add_argument('--num_epochs', type=int, help="The number of epochs to run", default=10)
parser.add_argument('--batch_size', type=int, help="The batch size", default=512)
parser.add_argument('--out', type=str, help="Output directory")
parser.add_argument('--device', type=str, help="The GPU to run on (e.g., cuda:0)")
parser.add_argument('--year_month', type=str, help="The year-month (YYYY-MM) of Reddit data to analyze",
                    default="2019-09")
parser.add_argument('--max_users', type=int, help="The maximum number of users to train on", default=-1)
parser.add_argument('--log_dir', type=str, help="Log directory for tensorboard",
                    default='/shared/0/projects/reddit-political-affiliation/working-dir/tensor-logs/')
parser.add_argument('--data_directory', type=str,
                    default='/shared/0/projects/reddit-political-affiliation/data/word2vec/data-mappings/')
parser.add_argument('--load_from', type=int, help='If load from a existing model', default=-1)
args = parser.parse_args()

embedding_dim = 50
batch_size = args.batch_size
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0}
EPOCHS = args.num_epochs
year_month = args.year_month
data_directory = args.data_directory + year_month

# User / subreddit data file
if args.network:
    network_path = args.network
else:
    # network_path = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + year_month + '.tsv'
    network_path = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/cong_train/'

if args.out:
    out_dir = args.out
else:
    out_dir = "/shared/0/projects/reddit-political-affiliation/data/word2vec/"

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
    device = torch.device("cuda:4")
    torch.cuda.set_device(0)

load_from = 9
