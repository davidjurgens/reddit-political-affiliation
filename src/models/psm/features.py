import pytz
from datetime import datetime

political_subreddits = [
    'Conservative',
    'democrats',
    'Republican',
    'Liberal',
    'GreenParty',
    'obama',
    'The_Donald',
    'Ask_Politics',
    'AskTrumpSupporters',
    'ChapoTrapHouse',
    'hillaryclinton',
    'SandersForPresident'
]

top_subreddits = [
    'funny',
    'AskReddit',
    'gaming',
    'aww',
    'Music',
    'pics',
    'science',
    'worldnews',
    'videos',
    'todayilearned',
    'movies',
    'news',
    'Showerthoughts',
    'EarthPorn',
    'gifs',
    'IAmA',
    'food',
    'askscience',
    'Jokes',
    'LifeProTips',
    'explainlikeimfive',
    'Art',
    'books',
    'mildlyinteresting',
    'nottheonion',
    'DIY',
    'sports',
    'blog',
    'space',
    'gadgets',
    'trees',
    'nba'
]

time_mappings = {
    "morning": 0,
    "afternoon": 1,
    "evening": 2,
    "night": 3
}


def get_time_of_day(created_utc):
    hour = datetime.fromtimestamp(int(created_utc), pytz.UTC).hour

    if 5 <= hour <= 11:
        return 'morning'
    if 12 <= hour <= 16:
        return 'afternoon'
    if 17 <= hour <= 20:
        return 'evening'

    return 'night'
