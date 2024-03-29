import glob
import re
import sys
from collections import *

sys.path.append('/home/kalkiek/projects/reddit-political-affiliation/')

from src.features.political_affiliations.comment_regexes import *
from src.data.data_helper import read_submissions, grab_bot_accounts


def parse_comment_affiliations(file_path, is_silver=False):
    if is_silver:
        dem_pattern, anti_rep_pattern, rep_pattern, anti_dem_pattern = DEM_PATTERN_SILVER, ANTI_REP_PATTERN_SILVER, \
                                                                       REP_PATTERN_SILVER, ANTI_DEM_PATTERN_SILVER
    else:
        dem_pattern, anti_rep_pattern, rep_pattern, anti_dem_pattern = DEM_PATTERN_GOLD, ANTI_REP_PATTERN_GOLD, \
                                                                       REP_PATTERN_GOLD, ANTI_DEM_PATTERN_GOLD
    bots = grab_bot_accounts()
    user_politics = defaultdict(list)
    counts = defaultdict(int)
    for submission in read_submissions(file_path):
        username, subreddit, created = submission['author'], submission['subreddit'], submission['created_utc']
        text = get_submission_text(submission)

        dem_match, rep_match, match = False, False, ""

        if re.findall(dem_pattern, text):
            match = re.findall(dem_pattern, text)[0][0]
            entry = {'politics': 'Democrat', 'match': match, 'match_type': 'dem', 'subreddit': subreddit,
                     'created': created, 'text': text}
            dem_match = True
        if re.findall(anti_rep_pattern, text):
            match = re.findall(anti_rep_pattern, text)[0][0]
            entry = {'politics': 'Democrat', 'match': match, 'match_type': 'anti_rep', 'subreddit': subreddit,
                     'created': created, 'text': text}
            dem_match = True
        if re.findall(rep_pattern, text):
            match = re.findall(rep_pattern, text)[0][0]
            entry = {'politics': 'Republican', 'match': match, 'match_type': 'rep', 'subreddit': subreddit,
                     'created': created, 'text': text}
            rep_match = True
        if re.findall(anti_dem_pattern, text):
            match = re.findall(anti_dem_pattern, text)[0][0]
            entry = {'politics': 'Republican', 'match': match, 'match_type': 'anti_dem', 'subreddit': subreddit,
                     'created': created, 'text': text}
            rep_match = True

        # Ignore comments that match both patterns
        if (dem_match or rep_match) and passes_cleanup_filters(username, bots, match, text, dem_match, rep_match,
                                                               counts):
            user_politics[username].append(entry)

    print(counts)
    print("File completed! Total political users found: {}".format(len(user_politics)))
    return user_politics


def passes_cleanup_filters(username, bots, regex_match, full_text, is_dem_match, is_rep_match, counts):
    # Ignore comments that match both patterns
    if is_dem_match and is_rep_match:
        counts['matches_both'] += 1
        return False

    # Ignore bots
    if username in bots:
        counts['bot_count'] += 1
        return False

    # Ignore comments with quote replies
    if "&gt;" in full_text:
        counts['quote_replies'] += 1
        return False

    # Ignore matches that are sarcastic or accusatory
    if re.findall(IS_ACCUSATION_PATTERN, full_text):
        counts['is_sarcastic'] += 1
        return False

    # Ignore matches that come inside of quotes
    if match_is_inside_quotes(regex_match, full_text):
        counts['is_inside_quotes'] += 1
        return False

    return True


def match_is_inside_quotes(regex_match, full_text):
    try:
        # Find the location of all quotations in the full text
        quote_indices = [m.start() for m in re.finditer('"', full_text)]

        # If there's an open quote at the end, drop it
        if len(quote_indices) % 2 == 1:
            quote_indices.pop()

        # If it comes after an open quote and before a closed one then it's a quotation
        regex_match_index = full_text.find(regex_match)
        it = iter(quote_indices)
        for index in it:
            start, stop = index, next(it)
            if start <= regex_match_index <= stop:
                return True
    except Exception as e:
        print(e)
    return False


def parse_name_from_filepath(filepath):
    # Get everything after the last /
    name = filepath.rsplit('/', 1)[-1]
    # Replace the extension with TSV
    return name.rsplit('.', 1)[0]


def user_politics_to_tsv(user_politics, out_file):
    print("Saving user politics to file: {}".format(out_file))
    with open(out_file, 'w') as f:
        for user, user_politics in user_politics.items():
            for entry in user_politics:
                if "match" in entry:
                    f.write(
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(user, entry['politics'], entry['match'],
                                                              entry['match_type'],
                                                              entry['subreddit'], entry['created'], entry['text']))
                else:
                    f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(user, entry['politics'], entry['match_type'],
                                                              entry['subreddit'], entry['created'], entry['text']))


def read_in_user_politics(in_files):
    user_politics = defaultdict(list)

    for in_file in in_files:
        print("Reading in user politics from file: {}".format(in_file))
        with open(in_file, 'r') as f:
            for line in f:
                try:
                    if len(line.split('\t')) == 7:
                        user, politics, match, match_type, subreddit, created, text = line.split('\t')
                        entry = {'politics': politics, 'match_type': match_type, 'subreddit': subreddit,
                                 'created': created.strip(), 'text': text}
                        user_politics[user].append(entry)
                    else:
                        user, politics, match_type, subreddit, created, text = line.split('\t')
                        entry = {'politics': politics, 'match_type': match_type, 'subreddit': subreddit,
                                 'created': created.strip(), 'text': text}
                        user_politics[user].append(entry)
                except Exception:
                    pass

    return user_politics


def get_submission_text(sub):
    text = ""
    if "body" in sub:
        text += sub['body'].lower()
    if "title" in sub:
        text += " " + sub['title'].lower()
    return " ".join(text.split())


def count_regex_matches(in_files):
    match_counter = Counter()
    user_politics = read_in_user_politics(in_files)
    for user, user_politics in user_politics.items():
        for entry in user_politics:
            match_counter[entry['regex_match']] += 1

    for k, v in match_counter.items():
        print("Regex pattern {} has {} matches".format(k, v))


def get_all_gold_users():
    gold_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/gold/*.tsv")
    gold_data = read_in_user_politics(gold_files)
    return set(gold_data.keys())


def get_all_silver_users():
    silver_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/silver/*.tsv")
    silver_data = read_in_user_politics(silver_files)
    return set(silver_data.keys())


if __name__ == '__main__':
    out = "/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/"

    files = glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.zst')
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.xz'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RC/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.bz2'))
    files.extend(glob.glob('/shared/2/datasets/reddit-dump-all/RS/*.xz'))

    for file in files:
        print("Starting on file: {}".format(file))
        user_politics = parse_comment_affiliations(file, is_silver=True)
        fname = parse_name_from_filepath(file)
        out_file = out + 'silver/' + fname + ".tsv"
        user_politics_to_tsv(user_politics, out_file)

    in_files = glob.glob("/shared/0/projects/reddit-political-affiliation/data/comment-affiliations/silver/*.tsv")
    count_regex_matches(in_files)
