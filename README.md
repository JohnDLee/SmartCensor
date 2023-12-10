# SmartCensor: A Transformer-base Detoxifier for toxic sentences
by John Lee and David Simonetti

In this project, we train a Transformer to detoxify sentences containing toxicity while attempting to preserve its meaning.

## Data

#### [Paradetox](https://huggingface.co/datasets/s-nlp/paradetox) - Parallel Detoxification Data
This is a dataset of 19,700 entries consisting of an original toxic sentence and a detoxified version of the same sentence intended to have the same meaning.
Example:
Obama has been a total failure , and now looks like a sore loser. ->
Obama has not been victorious.
We used this dataset to train our model as an instance of machine translation, i.e. our model was trained to translate a toxic sentence into a detoxified sentence. We also used 20% of this dataset as a dev set to validate the model during training.

#### [Jigsaw](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=test.csv.zip) - Wikipedia Toxic Comment Dataset
This dataset consists of comments taken from Wikipedia that have been labeled by human moderators as one of the following: toxic, severely toxic, obscene, threatening, insulting, or hateful of identity. We use roughly 16k test samples from this dataset to evaluate both the baseline and our trained model.

Example:
Chris, you mother fucker...all what you want to know about ChrisO you can find at www.ChrisO.homo.com'Bold text'" -> toxic, obscene, insult

#### [Toxic Word Bank](https://github.com/Orthrus-Lexicon/Toxic) - Dictionary of Toxic Words
This dataset is a curated list of known toxic words. It contains pure curse words, slurs, as well as common slangs and abbreviations of words used to harm others. We use this in our baseline in order to censor the toxic content of the sentence by searching for and removing  all words in this word bank.

## Setup



## Running


