import torch

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

print('Using PyTorch version:', torch.__version__, ' Device:', device)


import string
import unicodedata

# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


print(f"converting 'Ślusàrski' to {unicodeToAscii('Ślusàrski')}")


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    else:
        return allowed_characters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


print(f"The letter 'a' becomes {lineToTensor('a')}")  # notice that the first position in the tensor = 1
print(f"The name 'Ahn' becomes {lineToTensor('Ahn')}")  # notice 'A' sets the 27th index to 1


from io import open
import glob
import os
import time

from torch.utils.data import Dataset


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir  # for provenance of the dataset
        self.load_time = time.localtime  # for provenance of the dataset
        labels_set = set()  # set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        # read all the .txt files in the specified directory
        text_files = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in text_files:
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n')
            for name in lines:
                self.data.append(name)
                self.data_tensors.append(lineToTensor(name))
                self.labels.append(label)

        # Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item


alldata = NamesDataset("data/names")
print(f"loaded {len(alldata)} items of data")
print(f"example = {alldata[0]}")


train_set, test_set = torch.utils.data.random_split(
    alldata, [.85, .15],
    generator=torch.Generator(device='cpu').manual_seed(2024)
)

print(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")


import torch.nn as nn
import torch.nn.functional as F


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output


n_hidden = 128
rnn = CharRNN(n_letters, n_hidden, len(alldata.labels_uniq))
print(rnn)


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


input = lineToTensor('Albert')
output = rnn(input)  # this is equivalent to output = rnn.forward(input)
print(output)
print(label_from_output(output, alldata.labels_uniq))


import random
import numpy as np


def train(rnn, training_data, validation_data=None, n_epoch=10, n_batch_size=64, report_every=50,
          learning_rate=0.2, criterion=nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations
    and reporting thresholds
    """
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    val_losses = []
    val_accuracies = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad()  # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:  # for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))

        # --- Validation at the end of each epoch ---
        if validation_data is not None:
            val_loss = 0
            correct = 0
            total = 0
            rnn.eval()
            classes = validation_data.dataset.labels_uniq
            with torch.no_grad():
                for i in range(len(validation_data)):
                    (label_tensor, text_tensor, label, text) = validation_data[i]
                    output = rnn(text_tensor)
                    loss = criterion(output, label_tensor)
                    val_loss += loss.item()
                    guess, guess_i = label_from_output(output, classes)
                    label_i = classes.index(label)
                    if guess_i == label_i:
                        correct += 1
                    total += 1
            avg_val_loss = val_loss / len(validation_data)
            accuracy = correct / total * 100
            val_losses.append(avg_val_loss)
            val_accuracies.append(accuracy)
            rnn.train()

        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]:.4f}", end="")
            if validation_data is not None:
                print(f", val loss = {val_losses[-1]:.4f}, val acc = {val_accuracies[-1]:.2f}%")
            else:
                print()
        current_loss = 0

    return all_losses, val_losses, val_accuracies


start = time.time()
all_losses, val_losses, val_accuracies = train(
    rnn, train_set, validation_data=test_set,
    n_epoch=27, learning_rate=0.15, report_every=5
)
end = time.time()
print(f"training took {end - start}s")


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Figure 1: Training & Validation Loss Curves ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(all_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.grid(alpha=0.3)

# --- Figure 2: Validation Accuracy Curve ---
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy Curve')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))
    correct = 0
    total = 0

    rnn.eval()  # set to eval mode
    with torch.no_grad():  # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1
            if guess_i == label_i:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"Validation accuracy: {correct}/{total} = {accuracy:.2f}%")

    # Normalize by dividing every row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    # Set up plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy(), cmap='Blues')  # numpy uses cpu here so we need to use a cpu version
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Annotate each cell with the percentage value
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = confusion[i][j].item()
            if val > 0.5:
                ax.text(j, i, f'{val:.0%}', ha='center', va='center', color='white', fontsize=9)
            else:
                ax.text(j, i, f'{val:.0%}', ha='center', va='center', color='black', fontsize=9)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.show()


evaluate(rnn, test_set, classes=alldata.labels_uniq)
