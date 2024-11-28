from typing import Tuple

import torch
from torch import nn, Tensor


class CustomGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rz_gate = nn.Linear(input_size + hidden_size, hidden_size * 2)
        self.in_gate = nn.Linear(input_size, hidden_size)
        self.hn_gate = nn.Linear(hidden_size, hidden_size)
        ...

    def forward(self, xt: Tensor, ht0: Tensor):
        """
        xt: (batch_size, 1, input_size)
        ht0: (batch_size, hidden_size)
        """
        assert xt.shape[1] == 1, "xt should have shape (batch_size, 1, input_size)"
        assert ht0.shape[0] == xt.shape[0], "ht0 and ct0 should have shape (batch_size, hidden_size)"

        ht0 = ht0[:, None]  # (batch_size, 1, hidden_size)

        inputs = torch.cat([xt, ht0], -1)

        rz_gate_out = torch.sigmoid(self.rz_gate(inputs))
        r_gate_out, z_gate_out = torch.split(rz_gate_out, self.hidden_size, -1)  # Unsupported split axis !

        in_out = self.in_gate(xt)
        hn_out = self.hn_gate(ht0)
        n_gate_out = torch.tanh(in_out + r_gate_out * hn_out)

        ht = (1 - z_gate_out) * n_gate_out + z_gate_out * ht0
        return ht

    def set_weights(self, weight_ih_l0: Tensor, weight_hh_l0: Tensor, bias_ih_l0: Tensor, bias_hh_l0: Tensor):
        slice_index = self.hidden_size * 2
        rz_weight = torch.cat((weight_ih_l0[:slice_index], weight_hh_l0[:slice_index]), -1)
        rz_bias = bias_ih_l0[:slice_index] + bias_hh_l0[:slice_index]

        self.rz_gate.weight.data.copy_(rz_weight)
        self.rz_gate.bias.data.copy_(rz_bias)

        self.in_gate.weight.data.copy_(weight_ih_l0[slice_index:])
        self.in_gate.bias.data.copy_(bias_ih_l0[slice_index:])

        self.hn_gate.weight.data.copy_(weight_hh_l0[slice_index:])
        self.hn_gate.bias.data.copy_(bias_hh_l0[slice_index:])
        ...


class CustomGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.ModuleList(
            [CustomGRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )
        ...

    def forward(self, x: Tensor, states: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: (batch_size, 1, input_size)
        states: (num_layers, batch_size, hidden_size)
        """
        assert x.shape[1] == 1, "x should have shape (batch_size, 1, input_size)"
        assert states.shape[1] == x.shape[0], "states[0] should have shape (num_layers, batch_size, hidden_size)"

        batch_size = x.shape[0]
        state_list = torch.unbind(states, dim=0)  # (batch_size, hidden_size)
        gru_out, h_states_list = x, []
        for i in range(self.num_layers):
            gru_out = self.gru[i](gru_out, state_list[i])  # (batch_size, 1, hidden_size)
            h_states_list.append(gru_out.view(1, batch_size, self.hidden_size))
        h_states = torch.cat(h_states_list, dim=0)
        return gru_out, h_states

    def set_weights(self, state_dict: dict):
        assert len(state_dict) == 4 * self.num_layers, "state_dict should have 4 * num_layers keys"
        # TODO: more precise check for state_dict tensor shapes

        # initialize all weights and biases to zero
        for value in self.state_dict().values():
            value.data.zero_()

        keys = list(state_dict)
        for i in range(self.num_layers):
            self.gru[i].set_weights(*map(state_dict.get, keys[4 * i: 4 * (i + 1)]))
        ...


def main():
    # torch.manual_seed(1)
    batch_size, input_size, hidden_size, num_layers, time_steps = 257, 32, 64, 2, 100

    gru1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    gru2 = CustomGRU(input_size, hidden_size, num_layers)  # must be batch_first=True
    gru2.set_weights(gru1.state_dict())

    xt = torch.randn(batch_size, time_steps, input_size)
    ht1 = torch.zeros(num_layers, batch_size, hidden_size)
    ht2 = ht1.clone()

    out1_list, out2_list = [], []
    with torch.no_grad():
        for i in range(time_steps):
            x = xt[:, [i]]
            out1, ht1 = gru1(x, ht1)
            out2, ht2 = gru2(x, ht2)
            out1_list.append(out1)
            out2_list.append(out2)

    out1 = torch.cat(out1_list, dim=1)
    out2 = torch.cat(out2_list, dim=1)
    out3, _ = gru1(xt)

    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out1, out3)
    ...


if __name__ == "__main__":
    main()
    ...
