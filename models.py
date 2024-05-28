# Denis
# coding:UTF-8
from transformer import *


class Transmitter(nn.Module):
    def __init__(self, num_layers, vocab_size, key_size,
                 query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, num_units1, num_units2, **kwargs):
        super(Transmitter, self).__init__(**kwargs)
        self.transformer_encoder = TransformerEncoder(vocab_size, key_size, query_size, value_size, num_hiddens,
                                                      norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                                                      num_layers, dropout, **kwargs)
        self.dense1 = nn.Linear(num_hiddens, num_units1)
        self.relu1 = nn.LeakyReLU()
        self.dense2 = nn.Linear(num_units1, num_units2)
        self.relu2 = nn.LeakyReLU()

    def forward(self, X, valid_lens):
        outputs1 = self.transformer_encoder(X, valid_lens)
        outputs2 = self.relu1(self.dense1(outputs1))
        return self.relu2(self.dense2(outputs2))


class Receiver(nn.Module):
    def __init__(self, num_layers, vocab_size, key_size, query_size,
                 value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, channel_features, num_units, **kwargs):
        super(Receiver, self).__init__(**kwargs)
        self.dense1 = nn.Linear(channel_features, num_units)
        self.relu1 = nn.LeakyReLU()
        self.dense2 = nn.Linear(num_units, num_hiddens)
        self.relu2 = nn.LeakyReLU()
        self.transformer_decoder = TransformerDecoder(vocab_size, key_size, query_size, value_size, num_hiddens,
                                                      norm_shape, ffn_num_input, ffn_num_hiddens,
                                                      num_heads, num_layers, dropout, **kwargs)

    def forward(self, X, enc_outputs, enc_valid_lens):
        enc_output1 = self.relu1(self.dense1(enc_outputs))
        enc_output2 = self.relu2(self.dense2(enc_output1))
        dec_state = self.transformer_decoder.init_state(enc_output2, enc_valid_lens)
        output3, state = self.transformer_decoder(X, dec_state)
        return output3


class Transceiver(nn.Module):
    def __init__(self, num_layers, vocab_size, key_size,
                 query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, dropout, num_units1=256, num_units2=16, **kwargs):
        super(Transceiver, self).__init__(**kwargs)
        self.transmitter = Transmitter(num_layers, vocab_size, key_size, query_size,
                          value_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, dropout, num_units1, num_units2)
        self.receiver = Receiver(num_layers, vocab_size, key_size, query_size,
                        value_size, num_hiddens, norm_shape, ffn_num_input,
                        ffn_num_hiddens, num_heads, dropout, num_units2, num_units1)
        self.channel = Channels()

    def forward(self, enc_input, dec_input, valid_lens):
        enc_output = self.transmitter(enc_input, valid_lens)
        channel_output = PowerNormalize(self.channel.AWGN(enc_output, 0.1))
        pred = self.receiver(dec_input, channel_output, valid_lens)
        return pred, channel_output, enc_output




