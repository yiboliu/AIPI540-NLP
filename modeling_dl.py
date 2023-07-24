import copy
import time
import nltk
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch import nn

from data_processing import preprocess, select_words, select_words_for_each_sent
from model_struct import TextClassificationModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_datasets(df):
    train_iter = [(label, text) for label, text in zip(df['sentiment'].to_list(), df['selected_text'].to_list())]

    train_dataset = to_map_style_dataset(train_iter)

    num_train = int(len(train_dataset) * 0.8)
    num_val = len(train_dataset) - num_train
    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    return train_dataset, val_dataset


def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)


def build_vocab(data_iter):
    tokenizer = get_tokenizer('spacy')
    vocab = build_vocab_from_iterator(yield_tokens(data_iter, tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab, tokenizer


def collate_fn(batch, tokenizer, vocab):
    mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: mapping[x]
    label_list, text_list, offsets = [], [], [0]
    for (label, text) in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)

    return label_list.to(device), text_list.to(device), offsets.to(device)


def get_dataloader(dataset, batch_size, vocab, tokenizer):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=lambda x: collate_fn(x, tokenizer, vocab))


def build_dataloaders(df):
    train_dataset, val_dataset = build_datasets(df)
    vocab, tokenizer = build_vocab(train_dataset)
    batch_size = 128
    train_dataloader = get_dataloader(train_dataset, batch_size, vocab, tokenizer)
    val_dataloader = get_dataloader(val_dataset, batch_size, vocab, tokenizer)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    return dataloaders, dataset_sizes, vocab


def train_dl_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    model = model.to(device)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for labels, text, offsets in dataloaders[phase]:
                text = text.to(device)
                labels = labels.to(device)
                offsets = offsets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(text, offsets)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * text.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def launch_training_dl(dataloaders, dataset_sizes, vocab, model_path, vocab_path):
    num_classes = 3
    vocab_size = len(vocab)
    embed_dim = 64
    model = TextClassificationModel(vocab_size, embed_dim, num_classes).to(device)
    epochs = 100
    lr = 5.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2.0, gamma=0.9)
    model = train_dl_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=epochs)
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)


def serve_model_dl(model_path, vocab, input):
    mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
    model = TextClassificationModel(len(vocab), 64, 3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preprocessed = preprocess(input)
    words = select_words_for_each_sent(preprocessed, 2)
    nums = [vocab.get_stoi()[word] for word in words.split()]
    input = torch.tensor(nums).unsqueeze(0).to(device)
    output = model(text=input, offsets=None)
    return mapping[output.argmax().item()]


if __name__ == "__main__":
    df = select_words('train.csv')
    dataloaders, dataset_sizes, vocab = build_dataloaders(df)
    launch_training_dl(dataloaders, dataset_sizes, vocab, 'models/model-dl.pth', 'model/vocab.pkl')
