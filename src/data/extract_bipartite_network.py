import bz2
import glob
import json
import lzma
from collections import *
from os.path import basename

import zstandard as zstd
from tqdm import tqdm


def get_month_to_files(files):
    month_to_files = defaultdict(list)
    for fname in files:
        name = basename(fname)
        year_month = name.split('.')[0][3:]
        month_to_files[year_month].append(fname)
    print(len(month_to_files))
    return month_to_files


def get_file_type(fname):
    if fname.endswith('bz2'):
        return bz2
    elif fname.endswith('xz'):
        return lzma


def extract_bipartite_network(month_to_files, out_dir):
    months = list(month_to_files.keys())

    for month in tqdm(months):
        mfiles = month_to_files[month]
        user_to_subreddit_counts = defaultdict(Counter)
        for fname in tqdm(mfiles):
            if fname.endswith('.zst'):
                user_to_subreddit_counts = extract_zst_data(fname, user_to_subreddit_counts)
            else:
                user_to_subreddit_counts = extract_compressed_data(fname, user_to_subreddit_counts)

        save_network_to_tsv(out_dir + month + '.tsv', user_to_subreddit_counts)


def save_network_to_tsv(out_path, user_to_subreddit_counts):
    with open(out_path, 'wt') as outf:
        for user, scs in user_to_subreddit_counts.items():
            for sub, count in scs.items():
                outf.write(user + '\t' + sub + '\t' + str(count) + '\n')


def extract_compressed_data(fname, user_to_subreddit_counts):
    ftype = bz2 if fname.endswith('bz2') else lzma
    try:
        with ftype.open(fname, 'rt') as f:
            for line in f:
                try:
                    action = json.loads(line)  # post or comment
                    subreddit, user = action['subreddit'], action['author']
                    user_to_subreddit_counts[user][subreddit] += 1
                except KeyError:
                    pass
    except BaseException as e:
        print(repr(e))

    return user_to_subreddit_counts


def extract_zst_data(fname, user_to_subreddit_counts):
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fname) as reader:
        while True:
            chunk = reader.read(131072)
            if not chunk:
                break

            string_data = chunk.decode('utf-8')
            lines = string_data.split("\n")
            for i, line in enumerate(lines[:-1]):
                try:
                    action = json.loads(line)
                    subreddit, user = action['subreddit'], action['author']
                    user_to_subreddit_counts[user][subreddit] += 1
                except Exception:
                    pass

    return user_to_subreddit_counts


if __name__ == '__main__':
    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend('/shared/2/datasets/reddit-dump-all/RS/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))
