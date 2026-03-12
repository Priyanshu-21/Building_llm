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

'''
Layer Normalization: - Normalizing each element of matrices such that the mean ~ 0 and variance ~ 1. 
This helps the model to learn better during training, avoid exploding/vanishing gradient problem. 
'''
class LayerNormalization(nn.Module):
    def __init__(self, eplision: float = 10e-6):
        super().__init__()
        self.eps = eplision
        self.scale = nn.Parameter(torch.zeros([1])) # Trainable scale matrix 
        self.step = nn.Parameter(torch.ones([1])) # Trainable Step matrix 
    
    def forward(self, x):
        # Calculating mean and standard deviation for each element 
        mean = x.mean(dim= -1, keepdim= True)
        std = x.std(dim= -1, keepdim= True, unbiased= False)
        stand_x = (x - mean) / (std + self.eps)
        return self.scale * stand_x + self.step

'''
Feed Forward Network: - Layer_1 (batch_size, vocab_size, 4 * feature_dims)
Layer_2: Activation Function (Transformer Model:- RELU)
Layer_3: (batch_size, vocab_size, feature_dims)
'''
class RELU(nn.Module):
    def __init__(self):
        super().__init__() # calling nn.Module initialize method 
    
    def forward(self, x):
        return torch.relu(x)


class FeedForwardNetwork(nn.Module):
    def __init__(self, feature_dims: int, dropout: float):
        super().__init__()
        self.feature_dims = feature_dims
        # Stacking up layers together 
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dims, 4 * feature_dims, bias= True), # Expansion
            RELU(), # Activation
            nn.Linear(4 * feature_dims, feature_dims, bias= True), # Contraction
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.feed_forward(x))


'''
Multi-Head Attention: Self-Attention + Casual Attention for num_head > 1
num_head = number of attention head running parallel to compute context vectors 
head_dim = feature_dims / num_head 
'''
class MultiHeadAttention(nn.Module): 
    def __init__(self, feature_dims: int, num_head: int, dropout: float, qkv_bias: bool = False) -> None:
        super().__init__()
        self.feature_dims = feature_dims
        self.num_head = num_head
        assert feature_dims % num_head == 0, "feature_dims not divisible by num_head"
        # Calculation of head dimensions in the model 
        self.head_dims = feature_dims // num_head
        
        # Query, key, value: - Trainable matrices 
        self.W_q = nn.Linear(feature_dims, feature_dims, bias= qkv_bias)
        self.W_k = nn.Linear(feature_dims, feature_dims, bias= qkv_bias)
        self.W_v = nn.Linear(feature_dims, feature_dims, bias= qkv_bias)
        # To output layer
        self.W_o = nn.Linear(feature_dims, feature_dims, bias= qkv_bias)
        # Dropout layer 
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention_scores(query, key, value, mask, dropout: nn.Dropout):

        # Calculation of attention scores (query * key.T)
        # (batch_size, num_head, token, head_dim) * (batch_size, num_head, head_dim, token)
        # Result --> (batch_size, num_head, token, token)
        attention_score = query @ key.transpose(2, 3)

        # Masking Casual: tokens should attend only to previous tokens where mask values == 0 (True)
        if mask is not None:
            attention_score = attention_score.masked_fill_(mask== 0, 10e-8)
        
        # Calculating attention matrix: - softmax + Scaled dot product (sqrt(dims(key)))
        attention_matrix = torch.softmax(attention_score / key.shape[-1]**0.5, dim= -1)

        # Applying dropout 
        dropout(attention_matrix)

        # Calculating Context vectors 
        # (batch_size, num_head, token, token) @ (batch_size, num_head, token, head_dims)
        # (batch_size, num_head, token, head_dim) --> Transpose(1, 2) original config 
        context_vecs = (attention_matrix @ value).transpose(1, 2)

        # Returning attention_matrix to later visualize tokens 
        return context_vecs, attention_matrix

    # Slightly different approach for encoder and decoder block (Transformer Model)
    def forward(self, q, k, v, mask):
        
        # Have query, key, value matrix assigned for each token
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        # Transformation of query, key, value matrix (batch_size, tokens, num_head, head_dims)
        query = query.view(query.shape[0], query.shape[1], self.num_head, self.head_dims)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.head_dims)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.head_dims)

        # Changing position of matrix to group them num_head wise
        # (batch_size, token, num_head, head_dims) --> (batch_size, num_head, token, head_dims)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # call to attention static method 
        context_vecs, self.attention_matrix = MultiHeadAttention.attention_scores(query, key, value, mask, self.dropout)

        # Transforming context_vecs back to (batch_size, tokens, feature_dims)
        context_vecs = context_vecs.contiguous().view(context_vecs.shape[0], context_vecs.shape[1], self.num_head * self.head_dims)

        return self.W_o(context_vecs)


'''
Shortcut Connection: - x + sublayer(normalization(x))
This helps to prevent on vanishing gradient problem during model training
'''
class ShortConnection(nn.Module): 
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__() # calling to nn.Module init method
        self.dropout = nn.Dropout(dropout)
        # Calling to normalization layer 
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

'''
Encoder Block:- Encoder Layer * number of encoder blocks 
'''
class EncoderBlock(nn.Module):
    def __init__(self, multi_head_block: MultiHeadAttention, feed_forward_block: FeedForwardNetwork, dropout: float) -> None:
        super().__init__()
        # Calling multi-head attention block and feed forward block 
        self.multi_head_block = multi_head_block
        self.feed_forward_block = feed_forward_block

        # Residual/ Short Connection layers:- 2
        self.short_connection = nn.ModuleList(ShortConnection(dropout) for _ in range(2))
    
    def forward(self, x, src_mask):
        x = self.short_connection[0](x, lambda z: self.multi_head_block(x, x, x, src_mask))
        x = self.short_connection[1](x, lambda z: self.feed_forward_block(z))
        # x_1 = input + mha(norm(x)), x_2 = x_1 + ffd(norm(x)) 
        return x
    

# Encoder Main logic with heads = 6 (original transformer)
class Encoder(nn.Module):
    def __init__(self, encoder_layers: nn.ModuleList):
        super().__init__()
        # Calling encoder layer each time + doing normalization 
        #self.output_layer = []
        self.encoder_layers = encoder_layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        # In every encoder block connecting it together and normalization 
        for layer in self.encoder_layers:
            x = layer(x, mask)
            #self.output_layer.append(x)
        
        # Normalization 
        return self.norm(x)


# Testing out code functionality 
torch.manual_seed(123)
inputs = torch.randn([2, 3, 6])
feature_dims = inputs.shape[-1]
vocab_size = inputs.shape[1]

q = torch.randn([vocab_size, feature_dims])
k = torch.randn([vocab_size, feature_dims])
v = torch.randn([vocab_size, feature_dims])
mask = (torch.rand([vocab_size, vocab_size]) > 0.6).int()

# Change dims such as (batch_size, vocab_size, feature_dims)
q = torch.stack([q, q], dim= 0)
k = torch.stack([k, k], dim= 0)
v = torch.stack([v, v], dim= 0)
mask = torch.stack([mask, mask], dim=0)

# Encoder Block
'''mha = MultiHeadAttention(feature_dims, 2, 0.0)
print(f'Context_vecs \n{mha(q, k, v, mask)}')'''

encoder_layer = nn.ModuleList([
    EncoderBlock(
    MultiHeadAttention(feature_dims, 2, 0.0), 
    FeedForwardNetwork(feature_dims, 0.0), 
    0.0), 
    EncoderBlock(
    MultiHeadAttention(feature_dims, 2, 0.0), 
    FeedForwardNetwork(feature_dims, 0.0), 
    0.0), 
    EncoderBlock(
    MultiHeadAttention(feature_dims, 2, 0.0), 
    FeedForwardNetwork(feature_dims, 0.0), 
    0.0), 
    EncoderBlock(
    MultiHeadAttention(feature_dims, 2, 0.0), 
    FeedForwardNetwork(feature_dims, 0.0), 
    0.0), 
    EncoderBlock(
    MultiHeadAttention(feature_dims, 2, 0.0), 
    FeedForwardNetwork(feature_dims, 0.0), 
    0.0), 
    EncoderBlock(
    MultiHeadAttention(feature_dims, 2, 0.0), 
    FeedForwardNetwork(feature_dims, 0.0), 
    0.0), 
])

encoder = Encoder(encoder_layer)
print(f'EncoderBlock.Values: \n{encoder(inputs, mask)}') # Obj: - Why encoder block output is 1 for each element ?
