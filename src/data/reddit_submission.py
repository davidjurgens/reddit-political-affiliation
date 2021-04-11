class Submission:
    def __init__(self, submission_json):
        self.submission_json = submission_json
        # Core fields in both comments and posts
        self.username = submission_json['author']
        self.flair = submission_json['author_flair_text']
        self.subreddit = submission_json['subreddit']
        self.created = submission_json['created_utc']
        self.score = submission_json['score']
        self.controversiality = submission_json['controversiality']
        self.total_awards = submission_json['total_awards_received']
        self.gilded = submission_json['gilded']
        self.text = " ".join(submission_json['body'].split()).lower()

        self.num_comments = 0
        if self.is_post():
            self.num_comments = submission_json['num_comments']

    def is_post(self):
        return "num_comments" in self.submission_json

    def is_comment(self):
        return not self.is_post()
