class Submission:
    def __init__(self, submission_json):
        self.submission_json = submission_json
        # Core fields in both comments and posts
        self.username = submission_json['author']
        self.flair = submission_json['author_flair_text']
        self.subreddit = submission_json['subreddit']
        self.created = submission_json['created_utc']

        if 'score' in submission_json:
            self.score = int(submission_json['score'])
        else:
            self.score = 0

        if 'controversiality' in submission_json:
            self.controversiality = int(submission_json['controversiality'])
        else:
            self.controversiality = 0

        if 'total_awards_received' in submission_json:
            self.total_awards = int(submission_json['total_awards_received'])
        else:
            self.total_awards = 0

        if 'gilded' in submission_json:
            self.gilded = int(submission_json['gilded'])
        else:
            self.gilded = 0

        if 'body' in submission_json:
            self.text = " ".join(submission_json['body'].split()).lower()
        else:
            self.text = ""

        self.num_comments = 0
        if self.is_post():
            self.num_comments = int(submission_json['num_comments'])

        self.submission_type = "post" if self.is_post() else "comment"

    def is_post(self):
        return "num_comments" in self.submission_json

    def is_comment(self):
        return not self.is_post()

    def get_metadata_dict(self):
        return {'username': self.username, 'subreddit': self.subreddit, 'score': self.score,
                'submission_type': self.submission_type, 'gilded': self.gilded, 'created': self.created,
                'total_awards': self.total_awards, 'controversiality': self.controversiality}
