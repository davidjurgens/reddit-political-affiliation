class PoliticalComment:
    def __init__(self, comment_id, parent_id, username, subreddit, created, political_affiliation, text=""):
        self.comment_id = str(comment_id)
        self.parent_id = str(parent_id)
        self.username = str(username)
        self.subreddit = str(subreddit)
        self.created = str(created)
        self.political_affiliation = str(political_affiliation)
        self.text = str(text)

    def to_dict(self):
        return {"comment_id": self.comment_id, "parent_id": self.parent_id, "username": self.username,
                "subreddit": self.subreddit, "created": self.created, 'politics': self.political_affiliation,
                'text': self.text}

    def to_tsv_row(self):
        elements = [self.comment_id, self.parent_id, self.username, self.subreddit, self.created,
                    self.political_affiliation, self.text]
        return "\t".join(elements) + "\n"
