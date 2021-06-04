import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math


class RNN(torch.nn.Module):
    def __init__(self,n_features,seq_length, n_hidden, n_layers, output_length):
        super(RNN, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = n_hidden 
        self.n_layers = n_layers 
        self.output_length = output_length
    
        self.l_rnn = torch.nn.RNN(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)

        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, self.output_length)
        

    
    def forward(self, x):        
        batch_size, _, _ = x.size()
        
        rnn_out, self.hidden = self.l_rnn(x)
        x = rnn_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)

class LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length, n_hidden, n_layers, output_length):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = n_hidden 
        self.n_layers = n_layers 
        self.output_length = output_length
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, self.output_length)
        

    
    def forward(self, x):        
        batch_size, _, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x)
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)



def attention(query, key, value):
    a = torch.matmul(query, key.transpose(2,1).float())
    a /= torch.sqrt(torch.tensor(query.shape[-1]).float())
    a = torch.softmax(a , -1)
    
    return  torch.matmul(a,  value) 

class V(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(V, self).__init__()
        self.dim_val = dim_val        
        self.fc1 = nn.Linear(dim_input, dim_val, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
  
        return x

class K_Q(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(K_Q, self).__init__()
        self.dim_attn = dim_attn        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
    
    def forward(self, x):
        x = self.fc1(x)
        
        return x

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()

        self.value = V(dim_val, dim_val)
        self.key = K_Q(dim_val, dim_attn)
        self.query = K_Q(dim_val, dim_attn)
    
    def forward(self, x, kv = None):
        if(kv is None):
            return attention(self.query(x), self.key(x), self.value(x))
        
        return attention(self.query(x), self.key(kv), self.value(kv))
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        
        self.heads = nn.ModuleList(self.heads)
        
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
                      
        
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
            
        a = torch.stack(a, dim = -1) 
        a = a.flatten(start_dim = 2) 
        
        x = self.fc(a)
        
        return x
    


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]. squeeze(1)
        return x     
    


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.lin1 = nn.Linear(dim_val, dim_val)
        self.lin2 = nn.Linear(dim_val, dim_val)
        
        self.lay_norm1 = nn.LayerNorm(dim_val)
        self.lay_norm2 = nn.LayerNorm(dim_val)
    
    def forward(self, x):
        a = self.attn(x)
        x = self.lay_norm1(x + a)
        
        a = self.lin1(F.relu(self.lin2(x)))
        x = self.lay_norm2(x + a)
        
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.lin1 = nn.Linear(dim_val, dim_val)
        self.lin2 = nn.Linear(dim_val, dim_val)
        
        self.lay_norm1 = nn.LayerNorm(dim_val)
        self.lay_norm2 = nn.LayerNorm(dim_val)
        self.lay_norm3 = nn.LayerNorm(dim_val)
        
    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.lay_norm1(a + x)
        
        a = self.attn2(x, kv = enc)
        x = self.lay_norm2(a + x)
        
        a = self.lin1(F.relu(self.lin2(x)))
        
        x = self.lay_norm3(x + a)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers = 1, n_encoder_layers = 1, n_heads = 1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len
        
 
        self.encs = []
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads))
        
        self.decs = []
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads))
        
        self.pos = PositionalEncoding(dim_val)
        
        
        
        self.decs = nn.ModuleList(self.decs)
        self.encs = nn.ModuleList(self.encs)
        
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)
    
    def forward(self, x):
        embed_enc_x = self.pos(self.enc_input_fc(x))
        e = self.encs[0](embed_enc_x)
        for enc in self.encs[1:]:
            e = enc(e)
        

        embed_dec_x = self.dec_input_fc(x[:,-self.dec_seq_len:])
        d = self.decs[0](embed_dec_x, e)
        for dec in self.decs[1:]:
            d = dec(d, e)

        x = self.out_fc(d.flatten(start_dim=1))
        
        return x