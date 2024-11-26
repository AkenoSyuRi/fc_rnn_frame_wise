from typing import Tuple

import torch
from torch import nn, Tensor


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gate = nn.Linear(input_size + hidden_size, hidden_size * 4)
        ...

    def forward(self, xt: Tensor, ht0: Tensor, ct0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        xt: (batch_size, 1, input_size)
        ht0: (batch_size, hidden_size)
        ct0: (batch_size, hidden_size)
        """
        assert xt.shape[1] == 1, "xt should have shape (batch_size, 1, input_size)"
        assert ht0.shape[0] == xt.shape[0], "ht0 and ct0 should have shape (batch_size, hidden_size)"

        ht0 = ht0[:, None]  # (batch_size, 1, hidden_size)
        ct0 = ct0[:, None]  # (batch_size, 1, hidden_size)

        inputs = torch.cat([xt, ht0], -1)

        in_gate, forget_gate, cell_gate, out_gate = torch.split(self.gate(inputs), self.hidden_size, -1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        ct = forget_gate * ct0 + in_gate * cell_gate
        ht = out_gate * torch.tanh(ct)
        return ht, ct

    def set_weights(self, weight_ih_l0: Tensor, weight_hh_l0: Tensor, bias_ih_l0: Tensor, bias_hh_l0: Tensor):
        weight = torch.cat((weight_ih_l0, weight_hh_l0), -1)
        bias = bias_ih_l0 + bias_hh_l0

        self.gate.weight.data.copy_(weight)
        self.gate.bias.data.copy_(bias)
        ...


class CustomLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.ModuleList(
            [CustomLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )
        ...

    def forward(self, x: Tensor, states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        x: (batch_size, 1, input_size)
        states: 2 * (num_layers, batch_size, hidden_size)
        """
        assert x.shape[1] == 1, "x should have shape (batch_size, 1, input_size)"
        assert states[0].shape[1] == x.shape[0], "states[0] should have shape (num_layers, batch_size, hidden_size)"

        batch_size = x.shape[0]
        lstm_out, h_states_out, c_states_out = x, [], []
        for i in range(self.num_layers):
            ht0, ct0 = states[0][i], states[1][i]  # (batch_size, hidden_size)
            lstm_out, c_state = self.lstm[i](lstm_out, ht0, ct0)  # (batch_size, 1, hidden_size)
            h_states_out.append(lstm_out.view(1, batch_size, self.hidden_size))
            c_states_out.append(c_state.view(1, batch_size, self.hidden_size))
        ht = torch.cat(h_states_out, dim=0)
        ct = torch.cat(c_states_out, dim=0)
        return lstm_out, (ht, ct)

    def set_weights(self, state_dict: dict):
        assert len(state_dict) == 4 * self.num_layers, "state_dict should have 4 * num_layers keys"
        # TODO: more precise check for state_dict tensor shapes

        # initialize all weights and biases to zero
        for value in self.state_dict().values():
            value.data.zero_()

        keys = list(state_dict)
        for i in range(self.num_layers):
            self.lstm[i].set_weights(*map(state_dict.get, keys[4 * i : 4 * (i + 1)]))
        ...


def main():
    # torch.manual_seed(1)
    batch_size, input_size, hidden_size, num_layers, time_steps = 257, 32, 64, 2, 100

    lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    lstm2 = CustomLSTM(input_size, hidden_size, num_layers)  # must be batch_first=True
    lstm2.set_weights(lstm1.state_dict())

    xt = torch.randn(batch_size, time_steps, input_size)
    ht1 = torch.randn(num_layers, batch_size, hidden_size)
    ct1 = torch.randn(num_layers, batch_size, hidden_size)
    ht2 = ht1.clone()
    ct2 = ct1.clone()

    out1_list, out2_list = [], []
    with torch.no_grad():
        for i in range(time_steps):
            x = xt[:, [i]]
            out1, (ht1, ct1) = lstm1(x, (ht1, ct1))
            out2, (ht2, ct2) = lstm2(x, (ht2, ct2))
            out1_list.append(out1)
            out2_list.append(out2)

    out1 = torch.cat(out1_list, dim=1)
    out2 = torch.cat(out2_list, dim=1)
    torch.testing.assert_close(out1, out2)
    ...


if __name__ == "__main__":
    main()
    ...
