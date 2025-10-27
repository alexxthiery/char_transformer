# write a LSTM cell using jax/flax

import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple

class LSTMCell(nn.Module):
    """A single LSTM cell.

    Args:
        hidden_size: The number of features in the hidden state h.
    
    Input shape:  (B, input_size)
    Output shape: (B, hidden_size)
    """

    hidden_size: int

    @nn.compact
    def __call__(self, x, h_c):
        h, c = h_c  # Unpack hidden state and cell state
        input_size = x.shape[-1]

        # Concatenate input and hidden state
        concat = jnp.concatenate([x, h], axis=-1)

        # Compute gates
        gates = nn.Dense(4 * self.hidden_size)(concat)
        i, f, g, o = jnp.split(gates, 4, axis=-1)

        # Apply activations
        i = nn.sigmoid(i)  # input gate
        f = nn.sigmoid(f)  # forget gate
        g = jnp.tanh(g)    # cell candidate
        o = nn.sigmoid(o)  # output gate

        # Update cell state and hidden state
        new_c = f * c + i * g
        new_h = o * jnp.tanh(new_c)

        return new_h, (new_h, new_c)  # Return new hidden state and (h, c) tuple
    '''
        # Pre-LayerNorm -> MLP -> Residual add
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.hidden_size * 4)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_size)(h)
        return x + h  # Residual connection
        '''
    
class LSTM(nn.Module):
    """A multi-layer LSTM module.

    Args:
        hidden_size: The number of features in the hidden state h.
        num_layers: Number of LSTM layers.
    """

    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x, h_c):
        for i in range(self.num_layers):
            lstm_cell = LSTMCell(self.hidden_size)
            x, h_c = lstm_cell(x, h_c)
        return x, h_c  # Return final output and (h, c) tuple
    
# Example usage:
# lstm = LSTM(hidden_size=128, num_layers=2)
# h0 = jnp.zeros((batch_size, 128))
# c0 = jnp.zeros((batch_size, 128))
# output, (hn, cn) = lstm(input_data, (h0, c0))