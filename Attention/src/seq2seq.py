"""
基于RNN解码器的Seq2Seq模型实现（完整版）
包含简单DecoderRNN和带Bahdanau注意力的AttnDecoderRNN
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import time
import math
import os

# 脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 数据预处理 ==========
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    lines = open(os.path.join(SCRIPT_DIR, 'data', '%s-%s.txt' % (lang1, lang2)), encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

eng_prefixes = (
    "i am ", "i m ", "he is", "he s ", "she is", "she s ",
    "you are", "you re ", "we are", "we re ", "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def get_dataloader(batch_size):
    """创建 DataLoader，使用固定长度 padding"""
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, pairs

# ========== 编码器 ==========
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

# ========== 简单RNN解码器（无注意力） ==========
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        # 嵌入层: (batch, 1) -> (batch, 1, hidden_size)
        embedded = self.embedding(input)
        # GRU: (batch, 1, hidden_size) + (1, batch, hidden_size)
        #    -> (batch, 1, hidden_size) + (1, batch, hidden_size)
        output, hidden = self.gru(embedded, hidden)
        # 线性输出: (batch, 1, hidden_size) -> (batch, 1, output_size)
        output = self.out(output)
        return output, hidden

# ========== Bahdanau 注意力 ==========
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: (batch, 1, hidden_size) — 解码器当前隐藏状态
        # keys:  (batch, seq_len, hidden_size) — 编码器所有时间步输出

        # 1. 计算注意力分数: score = Va * tanh(Wa @ query + Ua @ keys)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # (batch, seq_len, 1)

        # 2. softmax 归一化
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # 3. 加权求和得到上下文向量
        context = torch.bmm(attn_weights.transpose(1, 2), keys)  # (batch, 1, hidden_size)

        return context, attn_weights

# ========== 带注意力的解码器 ==========
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)  # (1, batch, hidden) -> (batch, 1, hidden)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

# ========== 训练 ==========
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent) if percent > 0 else s
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=1, plot_every=1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder,
                          encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('[%s] Epoch %d/%d (%.0f%%) avg_loss=%.4f' % (
                timeSince(start, epoch / n_epochs),
                epoch, n_epochs, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses

# ========== 评估 ==========
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, input_lang, output_lang, n=10, test_pairs=None):
    if test_pairs is None:
        raise ValueError("test_pairs must be provided")
    for i in range(n):
        pair = test_pairs[i]
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        print('> 输入 (法文):', pair[0])
        print('= 目标 (英文):', pair[1])
        print('< 预测 (英文):', ' '.join(output_words))
        print()
    return test_pairs

# ========== 绘制损失曲线 ==========
def plot_training_curves(losses_simple, losses_attn):
    # 设置中文字体（避免方框乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    plt.figure(figsize=(10, 5))
    epochs = range(1, len(losses_simple) + 1)

    plt.plot(epochs, losses_simple, 'b-o', label='DecoderRNN (无注意力)', linewidth=2, markersize=6)
    plt.plot(epochs, losses_attn, 'r-s', label='AttnDecoderRNN (有注意力)', linewidth=2, markersize=6)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Seq2Seq 训练损失曲线对比', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(epochs)

    # 保存图片
    plot_path = os.path.join(SCRIPT_DIR, 'training_loss_curve.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n损失曲线已保存至: {plot_path}")

    plt.show()

# ========== 主程序 ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Seq2Seq 法语->英语翻译模型")
    print("="*60)

    # 超参数
    hidden_size = 64
    batch_size = 32
    n_epochs = 10

    # 准备数据
    print("\n[1] 准备数据...")
    input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size)
    print(f"词汇表大小: 法语={input_lang.n_words}, 英语={output_lang.n_words}")
    print(f"训练样本: {len(pairs)}")

    # ===== 训练简单 DecoderRNN =====
    print("\n" + "="*60)
    print("[A] 训练 简单RNN解码器 (DecoderRNN)")
    print("="*60)

    encoder_simple = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder_simple = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    print("\n--- 网络结构 ---")
    print(f"编码器 (EncoderRNN): {sum(p.numel() for p in encoder_simple.parameters()):,} 参数")
    print(encoder_simple)
    print(f"\n解码器 (DecoderRNN): {sum(p.numel() for p in decoder_simple.parameters()):,} 参数")
    print(decoder_simple)

    plot_losses_simple = train(train_dataloader, encoder_simple, decoder_simple, n_epochs)
    print(f"\n训练损失: {plot_losses_simple}")

    # 先随机抽取测试句（两个模型共用）
    n_test = 6
    test_pairs = [random.choice(pairs) for _ in range(n_test)]

    encoder_simple.eval()
    decoder_simple.eval()
    print("\n--- 简单RNN解码器 翻译结果 ---")
    evaluateRandomly(encoder_simple, decoder_simple, input_lang, output_lang, n=n_test, test_pairs=test_pairs)

    # ===== 训练带注意力的 AttnDecoderRNN =====
    print("\n" + "="*60)
    print("[B] 训练 注意力解码器 (AttnDecoderRNN)")
    print("="*60)

    encoder_attn = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder_attn = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    print("\n--- 网络结构 ---")
    print(f"编码器 (EncoderRNN): {sum(p.numel() for p in encoder_attn.parameters()):,} 参数")
    print(encoder_attn)
    print(f"\n注意力 (BahdanauAttention): {sum(p.numel() for p in decoder_attn.attention.parameters()):,} 参数")
    print(decoder_attn.attention)
    print(f"\n解码器 (AttnDecoderRNN): {sum(p.numel() for p in decoder_attn.parameters()):,} 参数")
    print(decoder_attn)

    plot_losses_attn = train(train_dataloader, encoder_attn, decoder_attn, n_epochs)
    print(f"\n训练损失: {plot_losses_attn}")

    encoder_attn.eval()
    decoder_attn.eval()
    print("\n--- 注意力解码器 翻译结果 ---")
    evaluateRandomly(encoder_attn, decoder_attn, input_lang, output_lang, n=n_test, test_pairs=test_pairs)

    # ===== 对比 =====
    print("\n" + "="*60)
    print("损失对比")
    print("="*60)
    print(f"DecoderRNN (无注意力) 最终损失: {plot_losses_simple[-1]:.4f}")
    print(f"AttnDecoderRNN (有注意力) 最终损失: {plot_losses_attn[-1]:.4f}")

    # ===== 绘制损失曲线 =====
    plot_training_curves(plot_losses_simple, plot_losses_attn)
