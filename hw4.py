import torch
import torchtext
import torch.nn as nn
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.optim as optim
import time
import warnings
warnings.filterwarnings('ignore')

# TEXT = data.Field(include_lengths=True)

# If you want to use English tokenizer from SpaCy, you need to install SpaCy and download its English model:
# pip install spacy
# python -m spacy download en_core_web_sm
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)

LABEL = data.LabelField(dtype=torch.float)
train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')

# TEXT.build_vocab(train_data)
# Here, you can also use some pre-trained embedding
TEXT.build_vocab(train_data,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text),
    batch_size=batch_size, device=device)

#Define the neural network
class LSTM(nn.Module):
    def __init__(self, in_dim, embed_dim, hidden_dim, out_dim, layer_count,
                 bidirectional, dropout, pad_index):
        super().__init__()

        self.embedding = nn.Embedding(in_dim, embed_dim, padding_idx=pad_index)

        self.rnn = nn.LSTM(embed_dim,
                           hidden_dim,
                           num_layers=layer_count,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))

        # pack sequence, length must be done on CPU
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'),enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # concatinate final forward and backward hidden layers and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)

IN_DIM = len(TEXT.vocab)
print(f'The LSTM vocab size is {IN_DIM:,}')
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
LAYER_COUNT = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_INDEX = TEXT.vocab.stoi[TEXT.pad_token]

model = LSTM(IN_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            LAYER_COUNT,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_INDEX)

def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'This model has {param_count(model):,} trainable parameters')

pretr_embeddings = TEXT.vocab.vectors

print(pretr_embeddings.shape)

model.embedding.weight.data.copy_(pretr_embeddings)

UNK_INDEX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_INDEX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_INDEX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

optimizer = optim.Adam(model.parameters())

crit = nn.BCEWithLogitsLoss()

model = model.to(device)
crit = crit.to(device)

def binary_acc(pred, label):
    rounded_preds = torch.round(torch.sigmoid(pred)) #closest integer
    correct = (rounded_preds == label).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, crit):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text

        preds = model(text, text_lengths).squeeze(1)

        loss = crit(preds, batch.label)

        acc = binary_acc(preds, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, crit):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            preds = model(text, text_lengths).squeeze(1)

            loss = crit(preds, batch.label)

            acc = binary_acc(preds, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start, end):
    elapsed = end - start
    mins = int(elapsed / 60)
    secs = int(elapsed - (mins * 60))
    return mins, secs


epochs = 10

best_valid_loss = float('inf')

for epoch in range(epochs):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, crit)
    valid_loss, valid_acc = evaluate(model, valid_iterator, crit)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

model.load_state_dict(torch.load('tut2-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, crit)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
