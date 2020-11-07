import torch
# from . import basic_model, register_model
# from .. import modules
import torch.nn as nn


# from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

# @register_model('sync_test')
# class sync_test(basic_model):
class sync_test(nn.Module):
    def __init__(self):
        super(sync_test, self).__init__(
            # 'sync_test',
            # embed_dim = model_config.embed_dim,
            # hidden_dim = model_config.hidden_dim,
        )
        self.linear1 = torch.nn.Linear(300, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 300)
        self.linear = nn.Sequential(
            nn.Linear(300, 10)
        )

    def forward(self, batch):
        source, wizard, target = batch
        vec = source.cuda()  # 这里加了.cuda()

        vec = self.linear1(vec)
        vec = torch.relu(vec)
        vec = self.linear2(vec)
        vec = torch.relu(vec)
        vec = self.linear3(vec)
        vec = torch.relu(vec)
        vec = self.linear4(vec)
        vec = torch.relu(vec)
        vec = self.linear(vec)

        return vec, None

    # @classmethod
    # def setup_model(cls):
    #     return cls
