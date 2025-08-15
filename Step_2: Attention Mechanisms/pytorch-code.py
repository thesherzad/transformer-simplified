# required libraries
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

### implmenent Input Embedding Layer ###
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # embedding matrix
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # forward pass
    def forward(self, x):
        # it's recommended to safeguard the input embedding by scaling it, to avoid getting overwhelmed by positional embedding
        # pass x through the embedding layer and scale it
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        return x

# sample token IDs (e.g., batch of 2 sequences, each with 3 tokens)
# example vocab: {0: [PAD], 1: "Transformer", 2: "Simplified", 3: "is", 4: "awesome"}
# don't confuse yourself with [PAD], we'll cover it in the future lessons
token_ids = torch.tensor([
    [1, 2, 3],  # "Transformer Simplified is"
    [4, 2, 0],  # "awesome Simplified [PAD]"
])

# instantiate input embedding layer
vocab_size = 5 # in real world, we can get the count of unique values of `token_ids`
embedding_dim = 4 # in real world scenario, you might use real embedding like 768 for GPT-2/BERT
embedding = InputEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim) 

embedded = embedding(token_ids)

embedded.shape

# view the result
print("Token IDs:\n", token_ids)
print("Input Embeddings:\n", embedded)

# output analysis:
# if you run this code (with no modification) on your machine, you'd get a tensor of `token size` times by `embedding dimension`
# token size: 6; 2 batches, each batch 3 tokens
# embedding dimension: 4

# the resulting output will be passed to the next step - that is Positional Encoding Layer.

### implement Positional Encoding Layer ### 
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super().__init__()

        # create matrix of zeros
        pe = torch.zeros(max_seq_len, embedding_dim)
        # create a column vector of token positions
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # compute frequencies used in the positional encoding formula
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embedding_dim))

        # assign positional encodings to even and odd dims
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        # register as buffer to not train it but still keep it with the model
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        # add pe of the correct length and return
        return x + self.pe[:, :seq_len, :]

# the embedding size has to be the same as the one used in Inpute Embedding
positional_encoding = PositionalEncoding(embedding_dim=embedding_dim, max_seq_len=32)
pos_encoded_output = positional_encoding(embedded)

pos_encoded_output.shape

print("Positional Embedding: \n", pos_encoded_output)

# output analysis
# we've 6 tokens, and embedding dimension of 4. So each token has 4 dimensions. Now, when we pass the output of Positional Encoding do the next layer, model will be able to gather information about tokens using their position and makes sense of the context of the given input.


### implement Attention Mechanisims ###

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # calculate head-dimension
        self.head_dim = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_length = x.size(1)
        # Split the input embeddings and permute
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    # Masking helps to avoid leakage or model cheating - we'll cover this in detail in Decoder layer later.
    def compute_attention(self, query, key, value, mask=None):
        # Compute scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        print("\nScores (before masking): ", scores[[0], [0]])
        # apply masking
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        # explore only the first time step
        print("\nScores: ", scores[[0], [0]])
        print("\nAttn Weight: ", attention_weights[[0], [0]])
        # apply attention to values
        return torch.matmul(attention_weights, value)

    def combine_heads(self, x, batch_size):
        # Combine heads back to (batch_size, seq_length, d_model)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Build the forward pass
        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)
        # Compute attention weight, combine heads and pass the result through linear layer
        attention_weights = self.compute_attention(query, key, value, mask)
        output = self.combine_heads(attention_weights, batch_size)
        return self.output_linear(output)

# Reproducible random seed
torch.manual_seed(42)

# Parameters
batch_size = 2
seq_length = 4
d_model = 8
num_heads = 2

# Create dummy input data
# Shape: (batch_size, seq_length, d_model)
query = torch.randn(batch_size, seq_length, d_model)
key   = torch.randn(batch_size, seq_length, d_model)
value = torch.randn(batch_size, seq_length, d_model)

# Optional mask: allow all positions to attend. I'll explain `masking` later in the Decoder layer step
mask = torch.ones(batch_size, 1, 1, seq_length)

# Create attention layer
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Forward pass
attn_output = mha(query, key, value, mask)

print("Query:\n", query)
print("\nKey:\n", key)
print("\nValue:\n", value)

# The `attn_output` is the sequence transformed by attention, where each token’s embedding now contains information from other tokens it “attended” to.
print("\nOutput:\n", attn_output)
attn_output.shape

