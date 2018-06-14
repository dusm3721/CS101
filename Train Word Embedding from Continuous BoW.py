import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)


##############################################
# Train Word Embeddings From Continuos BoW
##############################################

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word : i for i, word in enumerate(vocab)}

data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i-2], raw_text[i-1],
               raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])        

class CBOW(nn.Module):
    def __init__(self, context_size, embedding_dim, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(2*context_size*embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1,-1)
        logprobs = F.log_softmax(self.linear(embeds))
        return logprobs

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return autograd.Variable(torch.LongTensor(idxs))

losses = []
loss_function = nn.NLLLoss()
model = CBOW(CONTEXT_SIZE, EMBEDDING_DIM, vocab_size)
optimizer = optim.SGD(model.parameters(), lr=0.01)


x = make_context_vector(data[1][0], word_to_ix)

model(context_var)



for epoch in range(20):
    total_loss = torch.Tensor([0])
    for context, target in data:    
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        
        model.zero_grad()
        
        log_probs = model(context_var)
        
        y = autograd.Variable(torch.LongTensor([word_to_ix[target]]))
        loss = loss_function(log_probs, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
    losses.append(total_loss)
print(losses)        

w1 = autograd.Variable(torch.LongTensor([word_to_ix["we"]]))
w2 = autograd.Variable(torch.LongTensor([word_to_ix["We"]]))
w3 = autograd.Variable(torch.LongTensor([word_to_ix["with"]]))


v1 = model.embeddings(w1).data.numpy()[0,:]
v2 = model.embeddings(w2).data.numpy()[0,:]
v3 = model.embeddings(w3).data.numpy()[0,:]


c12 = np.dot(v1, v2) / (np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))
c13 = np.dot(v1, v3) / (np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v3,v3)))
c23 = np.dot(v3, v2) / (np.sqrt(np.dot(v3,v3))*np.sqrt(np.dot(v2,v2)))
