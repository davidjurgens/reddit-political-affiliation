Reddit Political Affiliation 
==============================


## Political Classification
_________________

### Behavioral Classifier
Analogous to training a word2vec model with separate user and subreddit embeddings which learn parameters to maximize if the user participates in the subreddit. We extend this approach to use
semi-supervised training in a multi-task setup: the traditional user2community model is retained and a separate linear layer is used to predict political affiliation from the user embedding if that user’s affiliation is known. This semi-supervised setup provides
structure to the user embeddings, ideally infusing all users with information on their affiliation based  on subreddit commenting behavior. Unlike the
text-based classifier, the behavioral model captures user engagement in politically-affiliated communities, even if the user never explicitly declares their
affiliation in comments. The primary difference between the behavioral and text classifiers is that the behavioral classifier captures whether a user associates with groups (subreddits) that are politically
affiliated (e.g., gun-rights or pro-life for conservative users), whereas the text classifier captures whether a user says something that reveals their politics.



**Code**
- Dataset: `src/data/word2vec` contains the code to build the bipartite network between users and subreddits
- Training `src/models/word2vec/train_model.py`

### Text Classifier

Some topics are politically oriented and can potentially reveal a user’s leaning,
e.g., discussing interests in gun rights. To infer affiliation from such statements, we train a RoBERTa model over comments made
from each user, excluding any statements they make that explicitly self-identify their affiliation.
The model predicts each comment, and we aggregate the model outputs by taking the mean of predictions for selected comments associated with a  user as the final label

**Code** `src/models/textclassifier`

**Data** `data/all_interaction_comment_ids.tsv`


### Username Classifier

Usernames can reveal aspects of identity e.g., Hillary4Prez reveals
a liberal leaning. To predict affiliation from names, we follow Wang and Jurgens (2018) and train a bidirectional character-based LSTM

**Code** `src/models/usernameclassifier`

**Data** Flairs used to build the dataset `src/features/political_affiliations/political_labels.py`


## Interactions

**Code** `src/models/interactions`

**Data** `data/all_interaction_comment_ids.tsv`

## Behavioral Analyses
_________________

### Bad Actors

**Code** `src/features/bad_actors`


### Predicting Flips

**Code** `src/models/psm`