# Implementation of transformer Model - Attention is all you need !
import torch 
import torch.nn as nn 
import math 

'''
Token/ Vector Embedding: - 
Converting each token in the vocab_size to have vector representation to higher dimensions to have semantic meaning with other tokens. 
'''
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, feature_dims: int):
        super().__init__() # Importing nn.Module class 
        self.feature_dims = feature_dims
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, feature_dims)

    # This is a trainable weight matrix layer 
    # Values of matrix will be trained/ optimized 
    def forward(self, x):
        return self.token_embedding(x) / math.sqrt(self.feature_dims)
    
'''
Positonal Encoding: - Using relative rule based encoding
Each token has it's positon defined to better understand the context of text provided in the input. 
'''
class PositionalEncoding(nn.Module):
    def __init__(self, context_length: int, feature_dims: int, dropout: float):
        super().__init__() # Importing nn.Module class
        self.context_length = context_length
        self.feature_dims = feature_dims
        # Creating skeleton of positional encoding matrix (fixed approximated values)
        # (context_legth, feature_dims)
        pe = torch.zeros(context_length, feature_dims)

        # Each token position in context length matrix 
        pos = torch.arange(0, context_length).unsqueeze(1) # Changing pos to colum wise [0, 1, 2, 3, ....]
        # freq_scale of each token 
        freq_scale = 2 * torch.arange(0, context_length) / feature_dims
        div_term = torch.pow(10000, freq_scale)

        # Setting up approximated values for position (even, odd)
        # Even position, using sine function 
        pe[:, 0::2] = torch.sin(pos * div_term)
        # Odd position, using cosine function 
        pe[:, 1::2] = torch.cos(pos * div_term)

        # Changing dimension of positional matrix to (batch_size, context_length, feature_dims)
        pe.unsqueeze(0)

        # Register positional encoding matrix (without trainable) to run on any device(cpu/gpu)
        self.register_buffer("positional_encoding", pe)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Input embedding = token Embedding + positional encoding 
        x = x + (self.positional_encoding[:, :x.shape[1]]).requires_grad_(False) # type: ignore # Not trainable matrix
        
        return self.dropout(x)


# Testing out code functionality 
inputs = torch.tensor([
    [0.43], # Your
    [0.55], # Journey
    [0.57], # Starts 
    [0.22], # With
    [0.77], # One
    [0.05], # Step
], dtype= torch.long)

example = TokenEmbedding(inputs.shape[0], 2)
pos = PositionalEncoding(inputs.shape[1], 2, 0.5)
print(f'TokenEmbedding.example: \n{example(inputs)}')
print(f'InputEmbedding.values: \n{pos(example(inputs))}')