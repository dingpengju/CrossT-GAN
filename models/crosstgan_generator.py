import torch
import torch.nn as nn
import os
from models.crosstgan import CrossTGAN

class CrossTGANGenerator(nn.Module):
    def __init__(self, args):
        super(CrossTGANGenerator, self).__init__()
        self.args = args

        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        self.crosstgan = CrossTGAN(args).to(args.device)

        if hasattr(args, 'pretrained_model') and args.pretrained_model:
            self.crosstgan.load_state_dict(torch.load(args.pretrained_model))
        
    def forward(self, x):
        b, seq, features = x.shape
        mask = torch.ones((b, seq, features), device=x.device)
        info = [self.args.data_id, seq, self.args.stride, "Detecting anomalies in time series data"]
        output = self.crosstgan(info, x, mask)
        fake_z = torch.zeros((b, seq, 10), device=x.device)

        return output, fake_z
