import torch
import os
import csv
import random
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import json

tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def build_vocabulary(data_path):
    train_data_path = os.path.join(data_path, 'train.csv')
    data_iter = []
    with open(train_data_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            label = int(row[0])
            text = row[1]
            data_iter.append((label, text))
    vocab = build_vocab_from_iterator(yield_tokens(data_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    word_map = {word: vocab[word] for word in vocab.get_itos()}

    with open(os.path.join(data_path, 'word_map.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(word_map, jsonfile, ensure_ascii=False, indent=4)

    return vocab


def sample_data_by_class(data, num_samples_per_class):
    class_data = {}
    for label, text in data:
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(text)

    sampled_data = []
    for label, texts in class_data.items():
        if len(texts) >= num_samples_per_class:
            sampled_texts = random.sample(texts, num_samples_per_class)
            for text in sampled_texts:
                sampled_data.append((label, text))

    return sampled_data


def process_and_save_data(data_path, vocab, data_type, num_samples_per_class):
    data_file_path = os.path.join(data_path, f'{data_type}.csv')
    all_data = []
    with open(data_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            label = int(row[0])
            text = row[1]
            all_data.append((label, text))

    sampled_data = sample_data_by_class(all_data, num_samples_per_class)

    unique_labels = sorted(set([label for label, _ in sampled_data]))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    processed_data = {'sents': [], 'labels': []}
    for label, text in sampled_data:
        tokenized_text = tokenizer(text)
        indexed_text = [vocab[token] for token in tokenized_text]
        processed_data['sents'].append(torch.tensor(indexed_text))
        processed_data['labels'].append(label_mapping[label])

    padded_sents = pad_sequence(processed_data['sents'], batch_first=True)
    processed_data['sents'] = padded_sents
    processed_data['labels'] = torch.tensor(processed_data['labels'])

    save_path = os.path.join(data_path, f'sents/{data_type.upper()}_data.pth.tar')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(processed_data, save_path)


def main():
    base_path = "Data/data/yahoo_answers_csv/"
    vocab = build_vocabulary(base_path)

    train_num_samples_per_class = 1000
    test_num_samples_per_class = 100

    process_and_save_data(base_path, vocab, 'train', train_num_samples_per_class)
    process_and_save_data(base_path, vocab, 'test', test_num_samples_per_class)


if __name__ == "__main__":
    main()