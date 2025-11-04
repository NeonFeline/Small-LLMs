import torch
import torch.nn as nn

class LSTMNetworkExplicit(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_lstm_layers):
        super(LSTMNetworkExplicit, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstms = nn.ModuleList([nn.LSTM(hidden_size, hidden_size, batch_first=True) for _ in range(num_lstm_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(num_lstm_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_lstm_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(0.1)
        ) for _ in range(num_lstm_layers)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_lstm_layers)])
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.num_lstm_layers = num_lstm_layers

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, hidden_state=None, return_hidden=False):
        out = self.embedding(x)
        hidden_states = []
        for i in range(self.num_lstm_layers):
            residual = out
            if hidden_state is not None:
                hc = hidden_state[i]
            else:
                hc = None
            out, hc = self.lstms[i](out, hc)
            out = self.dropouts[i](out)
            out = residual + out
            out = self.lns[i](out)
            residual = out
            out = self.mlps[i](out)
            out = residual + out
            out = self.lns2[i](out)
            hidden_states.append(hc)  
        out = self.output_proj(out)
        if return_hidden:
            return out, hidden_states
        return out