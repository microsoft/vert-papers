import torch.nn as nn

class ProjAdapter(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.w0 = nn.Linear(in_size, hidden_size, bias=True)
        self.w1 = nn.Linear(hidden_size, in_size, bias=True)
        self.act = nn.GELU()
        return

    def forward(self, input_hidden_states):
        hidden_states = self.act(self.w0(input_hidden_states))
        hidden_states = self.w1(hidden_states)
        return hidden_states + input_hidden_states

    
class AdapterStack(nn.Module):
    def __init__(self, adapter_hidden_size, num_hidden_layers=12):
        super().__init__()
        self.slf_list = nn.ModuleList([ProjAdapter(768, adapter_hidden_size) for _ in range(num_hidden_layers)])
        self.ffn_list = nn.ModuleList([ProjAdapter(768, adapter_hidden_size) for _ in range(num_hidden_layers)])
        return
    
    def forward(self, inputs, layer_id, name):
        if name == "slf":
            outs = self.slf_list[layer_id](inputs)
        else:
            assert name == "ffn"
            outs = self.ffn_list[layer_id](inputs)
        return outs