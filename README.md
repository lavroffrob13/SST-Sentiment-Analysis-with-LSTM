# SST-Sentiment-Analysis-with-LSTM

To run the code, cd into the directory where it is saved and enter "python hw4.py" into your command prompt. To install all necessary dependencies before running the code, input the following:
pip install nltk
pip install torchtext
pip install spacy
python -m spacy download en_core_web_sm

Hyperparameters such as dropout and number of LSTM layers can be adjusted using 74-79 in the code. EMBEDDING_DIM is the embedding dimension, HIDDEN_DIM is the hidden dimension, LAYER_COUNT is the number of LSTM layers, BIDIRECTIONAL is whether the LSTM should be bidirectional, and DROPOUT is the dropout regularization parameter.
