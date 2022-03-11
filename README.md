
## Overview

This repo contains the minimal code required to fit the model from the following paper:

`Li, J., Lin, Z., Fu, P., & Wang, W. (2021, November). Past, Present, and Future: Conversational Emotion Recognition through Structural Modeling of Psychological Knowledge. In Findings of the Association for Computational Linguistics: EMNLP 2021 (pp. 1204-1214).`

My main goal was to strip down the author's code to its minimal components, so that the logic and procedures were clear. The original code can be found here (https://github.com/LeqsNaN/SKAIG-ERC). I focused on the fitting the MELD dataset for the sake of simplicity, because each dataset in the paper required different model and preprocessing code.

Running the code achieves a test set F1-score close to that in the bottom righthand corner of Table 2. Even running their code did not yield the exact reported F1-score -- however, this is likely due to the fact that I needed to use a smaller batch size to fit on my GPU.

Scripts:
- The main script is `train_model.py`; This loads the conversation data and preprocessed features, builds the model, and then trains the model.
- I also wrote a short script `extract_features.py` to extract the commonsense features from COMET. The authors do not include this code, but only provide the feature as a .pkl file. I used a newer version of COMET, because I was more familiar with its interface.

Support code:
- `data.py` is responsible for additional data processing (e.g. tokenizing conversations) and provides a getter for individual data entries.
- `download_pretrained.py` is a short script I wrote for downloading the pretrained RoBERTa.
- `model.py` provides the code for the graph neural network, the utterance encoder, and the model class that combines them together.

Directories:
- `/results`: log files for the runs; log1.txt is using their features; log2.txt is using my features.
- `/pretrained`: location for pretrained RoBERTa and COMET (need to be downloaded)
- `/notebooks`: jupyter notebook for plotting the results.
- `/features`: conversation data, labels, and the edge features (extracted using COMET)
- `/comet`: interface for using comet. This code was taken from another project that I was working on.

Walk through plan:
(1) Figure 1 (data structure, goal, novelty of paper); Table 1 (MELD dataset); Figure 2 (general architecture)
(2) `train_model.py` main() and args()
(3) data_loaders() and `data.py`; more detailed structure of data.
(4) general structure of train() and evaluate()
(5) model details in `model.py`
(6) result logs, plotting train/dev/val F1-scores.

## Extra Notes
- It's not clear that the improvements over RoBERTa baseline are robust across random seeds and whether they are specific to the chosen hyperparameters. Hyperparameters include graph hidden state dimensionality, number of attention heads, how far forward/backward to connect each utterance to other utterances, number of layers in the GNN, whether to use residual connections, whether to map the edge representation, etc. Choices on these hyperparameters should be informed using the validation set.

## To-Do:
- Upload to github
- Run with larger batch size
- Run another seed.
