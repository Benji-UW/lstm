# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
embed_size = 128
hidden_size = 254
num_layers = 1
num_epochs = 5
batch_size = 100
seq_length = 10
learning_rate = 0.001

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('data/wiki.train.tokens', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length

test_ids = corpus.transform_to_ids("data/wiki.test.tokens")
dev_ids = corpus.transform_to_ids("data/wiki.valid.tokens")


# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

# Train the model
for epoch in range(num_epochs):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    
    for i in range(0, ids.size(1) - seq_length, seq_length):
        optimizer.zero_grad()
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)
        
        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        if True:# step % 100 == 0:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                    .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))
    # development set and training set performances
    for i in range(0, ids.size(1) - seq_length, seq_length):
        test_inputs = test_ids[:, i:i+seq_length].to(device)
        test_targets = test_ids[:, (i+1):(i+1)+seq_length].to(device)

        outputs, states = model(test_inputs, states)
        test_loss = criterion(outputs, test_targets.reshape(-1))
        print("TEST PERPLEXITY " + str(np.exp(loss.item())))

        dev_inputs = dev_ids[:, i:i+seq_length].to(device)
        dev_targets = dev_ids[:, (i+1):(i+1)+seq_length].to(device)

        outputs, states = model(dev_inputs, states)
        test_loss = criterion(outputs, dev_targets.reshape(-1))
        print("DEV PERPLEXITY " + str(np.exp(loss.item())))


    

# Save the model checkpoints
torch.save(model.state_dict(), 'model.ckpt')