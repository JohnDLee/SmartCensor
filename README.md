# SmartCensor: A Transformer-base Detoxifier for toxic sentences
by John Lee and David Simonetti

In this project, we train a Transformer to detoxify sentences containing toxicity while attempting to preserve its meaning.

## [Poster](results/NLP_Poster.pdf)

## [Paper]


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

The setup is rather involved.

```
1. Create a virtual environment
2. pip install -f requirements.txt (Some requirements, such as pytorch, may require manual installation depending on the machine)
3. git clone https://github.com/unitaryai/detoxify
4. pip install -e detoxify (For the Toxicity detection module)
5. source setup.sh (to initialize all paths)
```

## Running Code

### Baseline

#### Results
Baseline results are achieved from `python scripts/get_baseline.py <all|toxic|nontoxic>`. 

#### Demonstration
For a simple command line demonstration, run `python scripts/demo_baseline.py`. Input is taken from stdin and the detoxified output is printed back out.

### SmartCensor (Detoxifier)

#### Training
Training the detoxifier can be done from `python scripts/encoder_decoder.py train`. If cuda is not available, please swap to cpu in the code.

#### Results
Test results are achieved from `python scripts/encoder_decoder.py test -model <model_path>`. We have trained a model under `models_detoxifier/model_10.pt` that performs decently. It may require cuda to be loaded.

#### Demonstration
For a simple command line demonstration, run `python scripts/encoder_decoder.py demo -model <model_path>`. The same model as in results may be used.

### Toxifier
To be trained

#### Training
Training a toxifier can be done via `python scripts/encoder_decoder.py train -toxifier`.

#### Demonstration
For a simple command line demonstration, run `python scripts/encoder_decoder.py demo -model <model_path>`. `models_toxifier/model_10.pt` can be used.




