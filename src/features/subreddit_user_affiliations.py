import json
from itertools import combinations

from visualization.visualize import build_subreddit_commonality_graph


class Subreddit:
    def __init__(self, subreddit, author):
        self.subreddit_name = subreddit
        self.users = set(author)
        self.comment_count = 1

    def get_subreddit_user_overlap(self, other):
        return len(set.intersection(self.users, other.users))

    def __repr__(self):
        return self.subreddit_name
        # return "%s: %d users / %d comments" % (self.subreddit_name, len(self.users), self.comment_count)


class Overlap:
    def __init__(self, sub_one, sub_two, user_overlap):
        self.sub_one = sub_one
        self.sub_two = sub_two
        self.user_overlap = user_overlap

    def __repr__(self):
        return "%s and %s %d user overlap" % (self.sub_one, self.sub_two, self.user_overlap)


def parse_data(file_path, lines=1000):
    subreddits = []
    # Dictionary for O(1) lookup
    subreddit_indices = dict()

    with open(file_path) as f:
        for i in range(lines):
            line = json.loads(f.readline())
            subreddit_name, author = line["subreddit"], line["author"]

            if subreddit_name in subreddit_indices:
                index = subreddit_indices[subreddit_name]
                subreddits[index].users.add(line["author"])
                subreddits[index].comment_count += 1
            else:
                subreddits.append(Subreddit(subreddit_name, author))
                subreddit_indices[subreddit_name] = len(subreddits) - 1

    return subreddits


def get_all_subreddit_overlaps(subreddits):
    overlaps = []
    for combo in combinations(subreddits, 2):
        user_overlap = combo[0].get_subreddit_user_overlap(combo[1])
        if user_overlap > 0:
            overlap = Overlap(combo[0].subreddit_name, combo[1].subreddit_name, user_overlap)
            overlaps.append(overlap)

    return overlaps


# Helper functions

def get_top_overlaps(overlaps, n=25):
    sorted_overlaps = sorted(overlaps, key=lambda o: o.user_overlap, reverse=True)
    return sorted_overlaps[0:n]


def get_top_subreddits(subreddits, n=100):
    sorted_subreddits = sorted(subreddits, key=lambda s: s.comment_count, reverse=True)
    return sorted_subreddits[0:n]


if __name__ == '__main__':
    path = "/Users/kalkiek/Downloads/RC_2016-01"
    results = parse_data(path, lines=1000)
    ols = get_all_subreddit_overlaps(results)

    # Limit the results for visualization
    top_subreddits = get_top_subreddits(results, n=10)
    top_overlaps = get_top_overlaps(ols, n=len(ols))
    build_subreddit_commonality_graph(top_subreddits, top_overlaps)
