import torch
from tensorboardX import SummaryWriter

embedding_dim = 50
batch_size = 4000
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
EPOCHS = 15
year = '2018'
month = '05'

network_path = '/shared/0/projects/reddit-political-affiliation/data/bipartite-networks/' + year + '-' + month + \
               '_filtered.tsv'
flair_directory = "/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/20*.tsv"
out_dir = "/shared/0/projects/reddit-political-affiliation/data/word2vec/" + year + "-" + month + "/"

# Tensorboard
writer = SummaryWriter(logdir='/home/kalkiek/projects/reddit-political-affiliation/tensorboard-logs/')

device = torch.device("cuda:4")
