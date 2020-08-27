import ast
import glob
from collections import *
from os.path import basename

from tqdm import tqdm

from src.features.political__labels import labels


def save_flair_frequencies(month_files):
    flair_counts = Counter()

    for month, m_files in tqdm(month_files.items(), total=len(month_files)):
        for fname in m_files:

            with open(fname, 'rt') as f:
                for line in f:
                    cols = line[:-1].split('\t')
                    try:
                        sub_flairs = dict(ast.literal_eval(cols[1]))
                        for k, v in sub_flairs.items():
                            sub_flairs[k] = list(v)[0]
                    except:
                        pass

                    for sub, flair in sub_flairs.items():
                        flair_counts[flair] += 1

    for s, c in flair_counts.most_common(200):
        print(s, c)

    with open('/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/flairs-by-freq.tsv',
              'wt') as outf:
        for s, c in flair_counts.most_common():
            outf.write('%s\t%d\n' % (s, c))

    flair_to_label = {}
    with open('/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/flairs-by-label.tsv', 'rt') as f:
        for line in f:
            cols = line[:-1].split(',')
            flair_to_label[cols[0]] = cols[1]


def output_user_to_political_affiliations(month_files):
    output_dir = '/shared/0/projects/reddit-political-affiliation/data/flair-affiliations/'

    for month, mfiles in tqdm(month_files.items(), total=len(month_files)):
        user_to_political_affiliations = defaultdict(Counter)
        for fname in mfiles:

            with open(fname, 'rt') as f:
                for line in f:
                    cols = line[:-1].split('\t')
                    user = cols[0]
                    try:
                        sub_flairs = dict(ast.literal_eval(cols[1]))
                    except:
                        pass

                    for sub, flairs in sub_flairs.items():
                        for flair in flairs:
                            if flair not in labels:
                                continue
                            label = labels[flair]
                            user_to_political_affiliations[user][label] += 1

        print('Saw %d users with political affiliations in %s' % (len(user_to_political_affiliations), month))
        with open(output_dir + month + '.tsv', 'wt') as outf:
            for user, scs in user_to_political_affiliations.items():
                for label, count in scs.items():
                    outf.write(user + '\t' + label + '\t' + str(count) + '\n')


if __name__ == '__main__':
    files = glob.glob('/home/kalkiek/projects/reddit-political-affiliation/data/processed/*.tsv')

    month_to_files = defaultdict(list)
    for fname in files:
        name = basename(fname)
        year_month = name.split('.')[0][3:]
        month_to_files[year_month].append(fname)

    print("Months to files count: " + str(len(month_to_files)))

    output_user_to_political_affiliations(month_to_files)
    save_flair_frequencies(month_to_files)
