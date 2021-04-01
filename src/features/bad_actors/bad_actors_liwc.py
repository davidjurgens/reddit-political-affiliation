import sys
import re
import liwc
from collections import Counter

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.bad_actors.bad_actor_submissions import read_in_bad_actor_submissions


def bad_actors_bag_of_words(bad_actor_submission_file):
    words = []

    bad_actor_submissions = read_in_bad_actor_submissions([bad_actor_submission_file])

    for bad_actor, comments in bad_actor_submissions.items():
        for comment in comments:
            tokens = list(tokenize(comment))
            words.extend(tokens)

    return words


def tokenize(text):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)


if __name__ == '__main__':
    parse, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')

    bad_actor_words = bad_actors_bag_of_words(
        "/shared/0/projects/reddit-political-affiliation/data/bad-actors/submissions/flair-and-gold/RC_2019-09.tev")
    bad_actor_counts = Counter(category for token in bad_actor_words for category in parse(token))
    print(bad_actor_counts)
