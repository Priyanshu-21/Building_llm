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
    


# Testing out code functionality 
inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # Journey
    [0.57, 0.85, 0.65], # Starts 
    [0.22, 0.58, 0.33], # With
    [0.77, 0.25, 0.40], # One
    [0.05, 0.80, 0.55], # Step
], dtype= torch.long)

example = TokenEmbedding(inputs.shape[0], inputs.shape[-1])
print(f'TokenEmbedding.example: \n{example(inputs)}')