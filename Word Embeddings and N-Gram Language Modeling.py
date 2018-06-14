import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


##############################################
# Word Embeddings and N-Gram Language Modeling
##############################################

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigrams = [
            ([test_sentence[i], test_sentence[i+1]], test_sentence[i+2])
            for i in range(len(test_sentence)-2)
            ]
print(trigrams[:3])    

vocab = set(test_sentence)
word_to_ix = {word : i for i, word in enumerate(vocab)}
ix_to_word = {i : word for i, word in enumerate(vocab)}
   
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128) # 128 neurons in hidden layer
        self.linear2 = nn.Linear(128, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1,-1))
        z = F.relu(self.linear1(embeds))
        log_probs = F.log_softmax(self.linear2(z))
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:    
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
        
    
     

sentence = ["my", "proud"]
x = sentence
for i in range(20):
    x_var = autograd.Variable(torch.LongTensor([word_to_ix[word] for word in x]))
    y_var = model(x_var)
    max_y = -100
    max_ix = -1
    for i in range(len(y_var[0])):
        if y_var.data[0][i] > max_y:
            max_y = y_var.data[0][i]
            max_ix = i
    
    y = ix_to_word[max_ix]
    sentence.append(y)
    x = [x[1], y]            

poem = ""
for word in sentence:
    poem += word + " "
print(poem)

