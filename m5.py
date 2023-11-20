import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
from torchaudio.datasets import SPEECHCOMMANDS
import os
import logging

            
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

def pad_sequence(batch, max_len=-1): 
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t()[:max_len] for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    #print(batch.shape)
    return batch.permute(0, 2, 1)

class AudioClassifier:

    def __init__(self, train_set, val_set, get_label, batch_size=128, max_len=-1):
        self.get_label = get_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = M5().to(self.device)
        self.labels = sorted(list(set(self.get_label(datapoint) for datapoint in train_set)))
        print(self.labels)
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: self.collate(batch, max_len=max_len),
            pin_memory=True,
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda batch: self.collate(batch, max_len=max_len),
            pin_memory=True
        )
        self.transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=16000).to(self.device)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.0001)
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
        self.losses = []
        
    def collate(self, batch, max_len=-1):

        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number
        
        tensors, targets = [], []
        
        # Gather in lists, and encode labels as indices
        for item in batch:
            waveform = item[0]
            label = self.get_label(item)
            tensors += [waveform]
            targets += [self.label_to_index(label)]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors, max_len=max_len)
        targets = torch.stack(targets)

        return tensors, targets

    def label_to_index(self, word):
            # Return the position of the word in labels
            return torch.tensor(self.labels.index(word))


    def index_to_label(self, index):
            # Return the word corresponding to the index in labels
            # This is the inverse of label_to_index
            return self.labels[index]


    def train(self, epoch, log_interval):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):

            data = data.to(self.device)
            target = target.to(self.device)
            
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = F.nll_loss(output.squeeze(), target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print training stats
            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # record loss
            self.losses.append(loss.item())


    def validate(self, epoch):
        self.model.eval()
        correct = 0
        for data, target in self.val_loader:
                    
            data = data.to(self.device)
            target = target.to(self.device)
            
            # apply transform and model on whole batch directly on device
            data = self.transform(data)
            output = self.model(data)
                    
            pred = output.argmax(dim=-1)
            correct += number_of_correct(pred, target)

        print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(self.val_loader.dataset)} ({100. * correct / len(self.val_loader.dataset):.0f}%)\n")

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class CommonVoiceBinary:
    def __init__(self, start=0, end=-1):
        cv = torchaudio.datasets.COMMONVOICE(root="data/in/cv-corpus-15.0-2023-09-08/pl/", tsv="train.tsv")
        self.data = []
        if end == -1:
            end = len(cv)-1
        for i in range(start, end):
            try:
                if cv[i][2]['gender'] in ['male', 'female']:
                    self.data.append(cv[i])
            except IndexError:
                pass
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def majority(data):
    from collections import Counter
    maj, count = sorted(Counter([item[2]['gender'] for item in data]).items(), key=lambda x: x[1])[-1]
    return maj, count / len(data)
    
dataset = "CV"            
def main():
    logging.getLogger().setLevel(logging.INFO)
    if dataset == "SC":
        train_set = SubsetSC("training")
        val_set = SubsetSC("validation")
        get_label = lambda x: x[2]
    elif dataset == "CV":
        train_set = CommonVoiceBinary(start=0, end=10000)
        logging.info(f"Train set done: {len(train_set)}")
        logging.info(f"Train set majority = {majority(train_set)}")
        val_set = CommonVoiceBinary(start=10000, end=-1)
        logging.info(f"Val set done: {len(val_set)}")
        logging.info(f"Val set majority = {majority(val_set)}")
        
        get_label = lambda x: x[2]['gender']
    log_interval = 20
    n_epoch = 100
    net = AudioClassifier(train_set, val_set, get_label)
    for epoch in range(1, n_epoch + 1):
        net.train(epoch, log_interval)
        net.validate(epoch)
        net.scheduler.step()

main()
