import json
import torch.nn as nn
import torch
import os
from torchtext.vocab import build_vocab_from_iterator

class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        x = self.fc3(x)
        return x


class TextCNN_FE(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TextCNN_FE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        self.convs = nn.ModuleList([
                nn.Conv2d(
                    in_channels=1,
                    out_channels=100,
                    kernel_size=(size, emb_size)
                )
                for size in [3, 4, 5]
            ])
        self.relu = nn.ReLU()

    def forward(self, text):
        embeddings = self.embedding(text).unsqueeze(1)
        conved = [self.relu(conv(embeddings)).squeeze(3) for conv in
                  self.convs]
        pooled = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]
        flattened = torch.cat(pooled, dim=1)
        return flattened


class TextCNN(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_size):
        super(TextCNN, self).__init__()
        self.base = TextCNN_FE(vocab_size, emb_size)
        self.classifier = Classifier(300, n_classes)

    def forward(self, x):
        return self.classifier((self.base(x)))


def textcnn(n_classes):
    with open(os.path.join("../Data/data/yahoo_answers_csv/sents", 'word_map.json'), 'r') as j:
        word_map = json.load(j)
        vocab = build_vocab_from_iterator([word_map.keys()])
        vocab_size = len(word_map)
    return TextCNN(n_classes, vocab_size, 50)
