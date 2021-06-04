# Hyperparameters for Nasdaq 100 dataset 


TICKER = [
    "AMGN",
    "AAPL",
    "AMAT",
    "INTC",
    "PCAR",
    "PAYX",
    "MSFT",
    "ADBE",
    "CSCO",
    "XLNX",
    "QCOM",
    "COST",
    "SBUX",
    "FISV",
    "CTXS",
    "INTU",
    "AMZN",
    "EBAY",
    "BIIB",
    "CHKP",
    "GILD",
    "NLOK",
    "CMCSA",
    "FAST",
    "ADSK",
    "CTSH",
    "NVDA",
    "GOOGL",
    "ISRG",
    "VRTX",
    "HSIC",
    "BIDU",
    "ATVI",
    "ADP",
    "ROST",
    "ORLY",
    "CERN",
    "BKNG",
    "MU",
    "DLTR",
    "ALXN",
    "SIRI",
    "MNST",
    "AVGO",
    "TXN",
    "MDLZ",
    "ADI",
    "WDC",
    "REGN",
    "LBTYK",
    "VRSK",
    "NFLX",
    "MAR",
    "ILMN",
    "LRCX",
    "EA",
    "AAL",
    "WBA",
    "BMRN",
    "SWKS",
    "INCY",
    "MXIM",
    "TMUS",
    "EXPE",
    "TCOM",
    "ULTA",
    "CSX",
    "NTES",
    "MCHP",
    "CTAS",
    "KLAC",
    "HAS",
    "JBHT",
    "IDXX",
    "WYNN",
    "MELI",
    "ALGN",
    "CDNS",
    "SNPS",
    "ASML",
    "TTWO",
    "PEP",
    "XEL",
    "AMD",
    "NTAP",
    "VRSN",
    "LULU",
    "WLTW",
    "UAL",
]

start = '2010-01-01'
end = '2021-05-01'
interval='d'

window = 10

train_part = 0.7
val_part = 0.15

batch_size_train = 512
batch_size_val = 256
batch_size_test = 1

#Transformer
dec_seq_len = 1
enc_seq_len = window - dec_seq_len 

dim_val = 10
dim_attn = 64
n_heads = 3

n_decoder_layers = 3
n_encoder_layers = 3

lr_transformer = 1e-3
num_epochs_transformer = 35

#LSTM

n_hidden_lstm = 64
n_layers_lstm = 3

lr_lstm = 1e-3
num_epochs_lstm = 30

#RNN

n_hidden_rnn = 64
n_layers_rnn = 3

lr_rnn = 1e-3
num_epochs_rnn = 30 