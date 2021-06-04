# Hyperparameters for MOEX dataset 


TICKER = ['AFKS.ME', 'AFLT.ME', 'ALRS.ME', 'GAZP.ME', 'SBER.ME',  'ROSN.ME',  'PLZL.ME', 'TATN.ME', 'LKOH.ME', 
                  'NVTK.ME', 'GMKN.ME', 'SNGS.ME', 'NLMK.ME', 'CHMF.ME', 'ALRS.ME', 'MAGN.ME', 'PLZL.ME', 'MTSS.ME',
                  'VTBR.ME', 'MGNT.ME',  'IRAO.ME', 'PHOR.ME', 'POLY.ME', 'MTLR.ME', 'RSTI.ME', 'PHOR.ME']

start = '2014-01-01'
end = '2021-05-01'
interval='d'

window = 10

train_part = 0.65
val_part = 0.15

batch_size_train = 256
batch_size_val = 256
batch_size_test = 1

#Transformer
dec_seq_len = 1
enc_seq_len = window - dec_seq_len 

dim_val = 10
dim_attn = 128
n_heads = 3

n_decoder_layers = 1
n_encoder_layers = 2

lr_transformer = 1e-3
num_epochs_transformer = 35


#LSTM

n_hidden_lstm = 64
n_layers_lstm = 4

lr_lstm = 1e-3
num_epochs_lstm = 35

#RNN

n_hidden_rnn = 64
n_layers_rnn = 4 

lr_rnn = 1e-3
num_epochs_rnn = 30