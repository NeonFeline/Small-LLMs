class ResidualRNN(nn.Module):
    """
    RNN with residual connection that adds input projection to RNN output.

    Args:
        input_size: Size of input features
        hidden_size: Size of RNN hidden state
        output_size: Size of output features
        num_layers: Number of RNN layers (default: 1)
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(ResidualRNN, self).__init__()

        # Validate inputs
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("All sizes must be positive integers")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Project input to hidden_size for residual connection
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Final output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, last_hidden=None):
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            last_hidden: Previous hidden state of shape (num_layers, batch_size, hidden_size)
                        If None, initializes to zeros

        Returns:
            out: Output tensor of shape (batch_size, seq_len, output_size)
            hidden: Final hidden state of shape (num_layers, batch_size, hidden_size)
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq, features), got {x.dim()}D")

        batch_size, seq_len, feat_size = x.shape

        if feat_size != self.input_size:
            raise ValueError(
                f"Input feature size {feat_size} doesn't match expected {self.input_size}"
            )

        # Initialize hidden state if not provided
        if last_hidden is None:
            last_hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype
            )
        else:
            # Validate hidden state shape
            if last_hidden.shape != (self.num_layers, batch_size, self.hidden_size):
                raise ValueError(
                    f"Hidden state shape {last_hidden.shape} doesn't match expected "
                    f"({self.num_layers}, {batch_size}, {self.hidden_size})"
                )

        # RNN forward pass
        rnn_out, hidden = self.rnn(x, last_hidden)

        # Residual connection: project input and add to RNN output
        residual = self.input_projection(x)
        rnn_out_with_residual = rnn_out + residual

        # Final output projection
        out = self.fc(rnn_out_with_residual)

        return out, hidden


import torch
import torch.nn as nn

class RNNetworkExplicit(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_rnn_layers):
        super(RNNetworkExplicit, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.rnns = nn.ModuleList([nn.RNN(hidden_size, hidden_size, batch_first=True) for _ in range(num_rnn_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_rnn_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        ) for _ in range(num_rnn_layers)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_rnn_layers)])
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.num_rnn_layers = num_rnn_layers  # Added to access in forward

    def forward(self, x, hidden_state=None, return_hidden=False):
        out = self.input_proj(x)
        hidden_states = []
        for i in range(self.num_rnn_layers):
            residual = out
            if hidden_state is not None:
                h = hidden_state[i]
            else:
                h = None
            out, h = self.rnns[i](out, h)
            out = residual + out
            out = self.lns[i](out)
            residual = out
            out = self.mlps[i](out)
            out = residual + out
            out = self.lns2[i](out)
            hidden_states.append(h)
        out = self.output_proj(out)
        if return_hidden:
            return out, hidden_states
        return out
