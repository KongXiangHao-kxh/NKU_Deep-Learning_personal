import torch

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

print('Using PyTorch version:', torch.__version__, ' Device:', device)


import string
import unicodedata

# We can use "_" to represent an out-of-vocabulary character
allowed_characters = string.ascii_letters + " .,;'" + "_"
n_letters = len(allowed_characters)

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


print(f"converting 'Ślusàrski' to {unicodeToAscii('Ślusàrski')}")


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
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


print(f"The letter 'a' becomes {lineToTensor('a')}")
print(f"The name 'Ahn' becomes {lineToTensor('Ahn')}")


from io import open
import glob
import os
import time

from torch.utils.data import Dataset


class NamesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()

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


class LSTMCell(nn.Module):
    """手动的单层 LSTM 细胞，实现标准 LSTM 门控机制"""
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # 输入门 (i)
        self.W_ii = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)

        # 遗忘门 (f)
        self.W_if = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)

        # 细胞门 / 候选门 (g)
        self.W_ig = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=False)

        # 输出门 (o)
        self.W_io = nn.Linear(input_size, hidden_size, bias=True)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev, c_prev):
        """
        x:     (batch, input_size)
        h_prev: (batch, hidden_size)
        c_prev: (batch, hidden_size)
        returns: (h_next, c_next)  each (batch, hidden_size)
        """
        i = torch.sigmoid(self.W_ii(x) + self.W_hi(h_prev))
        f = torch.sigmoid(self.W_if(x) + self.W_hf(h_prev))
        g = torch.tanh(self.W_ig(x) + self.W_hg(h_prev))
        o = torch.sigmoid(self.W_io(x) + self.W_ho(h_prev))

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()

        self.hidden_size = hidden_size
        # 使用手动的 LSTMCell
        self.cell = LSTMCell(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        # line_tensor shape: (seq_len, batch, input_size)
        seq_len = line_tensor.size(0)
        batch = line_tensor.size(1)

        # 初始化隐藏状态和细胞状态为 0
        h = torch.zeros(batch, self.hidden_size, device=line_tensor.device)
        c = torch.zeros(batch, self.hidden_size, device=line_tensor.device)

        # 沿时间步迭代
        for t in range(seq_len):
            x_t = line_tensor[t]  # (batch, input_size)
            h, c = self.cell(x_t, h, c)

        # 取最后一个时间步的隐藏状态做分类
        output = self.h2o(h)
        output = self.softmax(output)

        return output


n_hidden = 128
lstm = CharLSTM(n_letters, n_hidden, len(alldata.labels_uniq))
print(lstm)


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i


input = lineToTensor('Albert')
output = lstm(input)
print(output)
print(label_from_output(output, alldata.labels_uniq))


import random
import numpy as np


def train(lstm, training_data, validation_data=None, n_epoch=10, n_batch_size=64, report_every=50,
          learning_rate=0.2, criterion=nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations
    and reporting thresholds
    """
    current_loss = 0
    all_losses = []
    val_losses = []
    val_accuracies = []
    lstm.train()
    optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        lstm.zero_grad()

        # create some minibatches
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) // n_batch_size)

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = lstm.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(lstm.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))

        # --- Validation at the end of each epoch ---
        if validation_data is not None:
            val_loss = 0
            correct = 0
            total = 0
            lstm.eval()
            classes = validation_data.dataset.labels_uniq
            with torch.no_grad():
                for i in range(len(validation_data)):
                    (label_tensor, text_tensor, label, text) = validation_data[i]
                    output = lstm(text_tensor)
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
            lstm.train()

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
    lstm, train_set, validation_data=test_set,
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
plt.title('LSTM Loss Curves')
plt.legend()
plt.grid(alpha=0.3)

# --- Figure 2: Validation Accuracy Curve ---
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('LSTM Validation Accuracy Curve')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


def evaluate(lstm, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))
    correct = 0
    total = 0

    lstm.eval()
    with torch.no_grad():
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = lstm(text_tensor)
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
    cax = ax.matshow(confusion.cpu().numpy(), cmap='Blues')
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


evaluate(lstm, test_set, classes=alldata.labels_uniq)
