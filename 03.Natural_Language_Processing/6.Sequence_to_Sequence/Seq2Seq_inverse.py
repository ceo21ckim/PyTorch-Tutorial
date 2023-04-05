# generate random alphabet sequence
from unittest.util import _MAX_LENGTH
import numpy as np 

import torch 
import torch.nn as nn 
from torch.utils.data import Dataset 
import string

MAX_LENGTH = 15
def generate_random_alphbet_index(MAX_LENGTH=MAX_LENGTH):
    random_length = np.random.randint(10, MAX_LENGTH-2)
    random_alphabet_index = np.random.randint(0, 26, random_length) + 3 
    return random_alphabet_index.tolist(), random_length 

class AlphabetDataset(Dataset):
    def __init__(self, n_dataset = 1000):
        bos = 0 
        eos = 1 
        pad = 2
        self.inputs = []
        self.labels = []
        self.length = []
        
        for _ in range(n_dataset):
            # make input example 
            aindex, alen = generate_random_alphbet_index()

            iindex = aindex[::-1]
            
            # add bos, eos and pad 
            n_pad = MAX_LENGTH - len(aindex) - 1 
            aindex = aindex + [eos] + [pad]*n_pad 
            iindex = iindex + [eos] + [pad]*n_pad 
            
            self.inputs.append(aindex)
            self.labels.append(iindex)
            self.length.append(alen)
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return [ 
                torch.tensor(self.inputs[index], dtype=torch.long), 
                torch.tensor(self.labels[index], dtype=torch.long), 
                torch.tensor(self.length[index], dtype=torch.long)
                ]

train_dataset = AlphabetDataset(n_dataset = 3000)
valid_dataset = AlphabetDataset(n_dataset = 300)


# generate DataLoader for alphabet sequence inversing data 
import time
import torch 

import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader

def collate_fn(batch):
    inputs = pad_sequence([b[0] for b in batch], batch_first=True)
    targets = pad_sequence([b[1] for b in batch], batch_first=True)
    lengths = torch.stack([b[2] for b in batch])
    
    lengths, indice = torch.sort(lengths, descending=True)
    inputs = inputs[indice]
    targets = targets[indice]
    return inputs, targets, lengths
    
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=16)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=1)

dic1 = {'a':1}
dic2 = {'b':2}

def i2a(index):
    result = [ ]
    dic1 = {'bos' : 0, 'eos' : 1, 'pad':2 }
    dic2 = {alpha:(idx+3) for idx, alpha in enumerate(string.ascii_lowercase)}
    dic1.update(dic2)
    for idx in index: result.append(dic1[idx])
    return result 
    

# define Sequence to Sequence

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        
    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden
    
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.dropout_p = dropout_p
        self.max_length = max_length 
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_fisrt=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
        
    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs) # (B, 1, H)
        embedded = self.dropout(embedded)
        
        # query: embedded 
        # key: hidden 
        # value: encoder_outputs 
        
        attn_weights = F.softmax(
            self.attn(
                torch.cat((embedded, hidden.transpose(0, 1)), -1) # (B, 1, 2H)
            ),
            dim=-1
        )
        
        attn_applied = torch.bmm(attn_weights, encoder_outputs) # (B, 1, H)
        
        output = torch.cat((embedded, attn_applied), -1)
        
        output = self.attn_combine(output)
        
        output = F.relu(output)
        
        output, hidden = self.gru(output, hidden)
        
        output = F.log_softmax(self.out(output), dim =-1)
        
        return output, hidden, attn_weights 
    
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device='cuda:0' if torch.cuda.is_available() else 'cpu')


# model training 
bos = 0
eos = 1 
pad = 2
teacher_forcing_ratio = 0.5
device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, 
          criterion, max_length=MAX_LENGTH, with_attention=True):
    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.initHidden(batch_size)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    decoder_input = torch.tensor([bos]*batch_size, device=device)
    decoder_input = decoder_input.unsqueeze(-1)
    
    decoder_hidden = encoder_hidden 
    
    use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
    
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, 
            decoder_hidden, 
            encoder_outputs 
        )
        decoder_output = decoder_output.squeeze(1)
        
        loss += criterion(decoder_output, target_tensor[:, di])
        
        if use_teacher_forcing:
            decoder_input = target_tensor[:, di].unsqueeze(-1)
        
        else:
            topv,topi = decoder_output.topk(1)
            decoder_input = topi 
            
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length 

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, with_attention=True):
    start = time.time()
    plot_losses = []
    print_loss_total = 0 
    plot_loss_total = 0 
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    for iter, batch in enumerate(train_dataloader):
        input_tensor, target_tensor, length_tensor = batch 
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        length_tensor = length_tensor.to(device)
        
        loss = train(input_tensor, target_tensor, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, criterion, with_attention=with_attention)

        print_loss_total += loss 
        plot_loss_total += loss 
        
        if (iter+1) % print_every == 0 :
            print_loss_avg = print_loss_total / print_every 
            print_loss_total = 0 
            print(print_loss_avg)
        
        if (iter+1) % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every 
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0 
            

# evaluate 
def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():
        batch_size = input_tensor.size(0)
        encoder_hidden = encoder.initHidden(batch_size)
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)
        
        decoder_input = torch.tensor([bos]*batch_size, device=device)
        decoder_input = decoder_input.unsqueeze(-1)
        decoder_hidden = encoder_hidden 
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, 
                decoder_hidden, 
                encoder_outputs
            )
            decoder_output = decoder_output.squeeze(1)
            decoder_attention[:, di] = decoder_attention 
            
            topv, topi = decoder_output.topk(1)
            decoder_input = topi 
            if topi.item() == eos:
                decoded_words.append('</s>')
                break 
            else:
                decoded_words.append(i2a(topi.item()))
            
            
        return decoded_words, decoder_attentions[:di + 1]
