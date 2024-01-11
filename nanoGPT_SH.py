import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.onnx
import time 
import json
#from torch.quantization import quantize_dynamic, CalibrationObserver, default_qconfig

timed = time.time()

#hyperparameters
batch_size = 64 #64 #32 #independant sequences processed in parallel
block_size = 256 #64#256 #8 #max context length for predictions 
max_iter = 20000 #5000 #20000 #steps
eval_interval = 500 #300 
lr = 3e-4 #1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200 # steps for evaluation
n_embd = 384 #128 #384 #32 #batch_size * n_head (and not n_layer)  
n_head = 6 #4 #6 
n_layer = 6 #6
dropout = 0.2
#--------------------------------------------------------------

torch.manual_seed(1)
with open("superheroes_nlp_dataset.csv", 'r', encoding = 'utf-8') as f:
    text = f.read()

chars1 = set(text)
chars2 = list(set(text))
chars = sorted(list(set(text)))
#print("chars1: ", chars1, " chars2: ", chars2, " chars3: ", chars, sep = "\n")
vocab_size = len(chars)

#create a mapping from str to int
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

# print(stoi)
# exit()

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l]) 

#train and test splits
data = torch.tensor(encode(text), dtype= torch.long)
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

#data loading
def get_batch(split):
    data = train_data if split == "train" else validation_data
    ix =  torch.randint(len(data) - block_size, (batch_size,)) #why batch_size?
    x = torch.stack([data[i: i + block_size] for i in ix]) #for context: Concatenates a sequence of tensors along a new dimension
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]) #for target
    x, y = x.to(device), y.to(device)
    return x, y

# estimate the loss ...
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "validation"]:
        losses = torch.zeros(eval_iters) # is eval_iters = len(train_data) or len(validation_data) (depending on split) or smthg else??
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# self attention model

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #B, T, C
        q = self.query(x) #B, T, C
        # compute attention scores ("affinities between words")
        wei = q @ k.transpose(-2, -1) * C **-0.5 # => B, T, T
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) #B, T, T
        wei = F.softmax(wei, dim = -1) # B, T, T
        wei = self.dropout(wei) # B, T, T
        #perform the weighted agreggation of the values #??
        v = self.value(x) # B, T, C
        out = wei @ v # B, T, C
        return out

class MultiHeadAttention(nn.Module): #multiple heads of self-attention in parallel
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        #self.linear = nn.Linear(n_heads*n_embd, n_embd) #what for?
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) #B, T, n_heads*C
        out = self.dropout(self.proj(out)) #B, T, C
        return out

#MLP
class FeedForward(nn.Module): # a simple linear layer followed by a non-linearity
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(), #pq ces deux couches et pas plus?
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module): #transformer block (communication followed by computation)
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) 
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Bigram Model

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #n_embd?
        self.position_embedding_table = nn.Embedding(block_size, n_embd) #??
        #self.sa_head = Head(n_embd) #
        self.Blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        #self.sa_head = MultiHeadAttention(4, n_embd//4) #4 heads of 8-dimensional self-attention 
        #self.ffwd = FeedForward(n_embd)
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) #??

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) #weights? prediction? #?? #B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device= device)) # T, C
        x = tok_emb + pos_emb # B, T, C
        x = self.Blocks(x) #apply the transformer blocks (B, T, C)
        #x = self.sa_head(x) #apply one head of self-attention (B, T, C)
        #x = self.ffwd(x) #apply the feed-forward network (B, T, C)
        logits = self.lm_head(x) #?? #B, T, vocab_size

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens): #max_new_tokens = len(data) - block_size?
        for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # on prend tout dans B et C et que le dernier élément de T (contexte)ie la target
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples= 1)
            idx = torch.cat((idx, idx_next), dim = 1) 
        return idx

m = BigramLanguageModel()
model = m.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, "M parameters") 

# create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

best_validation_loss = float('inf')

for iter in range(max_iter+1): #?? max_iter??
    if iter % eval_interval == 0: #?? eval_interval?
        losses = estimate_loss()
        #print(f"iteration: {iter}, losses: {losses}")
        print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['validation']:.4f}")

        if losses['validation'] < best_validation_loss:
            best_validation_loss = losses['validation']
            torch.save(model.state_dict(), "best_model.pth")
            #print("Best model saved.")

    # now showtime (bleak stuff I don't understand yet)
    #sample a batch of data and evaluate the loss
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none= True) #??
    loss.backward() # I forgot the meaning of that
    optimizer.step() 


#generate from the model 
context = torch.zeros((1, 1), dtype = torch.long, device = device) #??
print("RASRASRASRASRASRASkxjfvdjsmhÏgnvsfjvl¬ƒÒ≠vfskv,sdl:kvdxnv")
print("")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist())) #max_new_tokens = combien on veut générer de lettres out of the blue?
 #[0] car [1]: device = blablabla donc préciser qu'on parle du tenseur avec les valeurs

timef = time.time()
time_total = timef - timed

# Calculate hours and minutes
hours = int(time_total // 3600)
minutes = int((time_total % 3600) // 60)

print("Time: {} hours, {} minutes".format(hours, minutes))


best_model = BigramLanguageModel()
best_model.load_state_dict(torch.load("best_model.pth"))
best_model = best_model.to(device)
best_model.eval()


# Create a dummy input tensor (you might need to adjust the shape and type)
dummy_input = torch.zeros((1, 256), dtype=torch.long).to(device)

# Export the model to ONNX
torch.onnx.export(
    best_model,
    dummy_input,
    "model.onnx",
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size', 1: 'sequence_length'}}
)


# quantization steps
#model = torch.load('your_model.pth')  # or define your model

# Choose the quantization configuration
#qconfig = default_qconfig

# Perform dynamic quantization
#quantized_model = quantize_dynamic(
    #model, qconfig=qconfig,
    #dtype=torch.qint8,  # You can choose other data types like torch.qint8, torch.quint8, etc.
#)

#scripted_quantized_model = torch.jit.script(quantized_model)
#scripted_quantized_model.save('quantized_model_scripted.pt')
