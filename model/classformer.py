import torch
from torch import nn

from model.embeding import PatchEmbedding
from model.encoder_decoder import EncoderDecoder


class Classformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = PatchEmbedding()
        self.encoder_decoder = EncoderDecoder(self.embedding.patch_size**2*3)
        self.classifier = Classifier(self.encoder_decoder.c_out)

    def forward(self, img):
        x = self.embedding(img)
        x_en = self.encoder_decoder(x)
        pred = self.classifier(x_en)
        # pred = torch.argmax(y, dim=-1)
        return pred


class Classifier(nn.Module):
    def __init__(self, c_in=2048, c_out=2):
        super().__init__()
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.mlp = torch.nn.Linear(c_in, c_out)
        self.act = torch.nn.Softmax(dim=1)
        return

    def forward(self, x):
        y = self.pooling(x.permute(0, 2, 1).contiguous())
        y = y.permute(0, 2, 1).contiguous()
        y = y.squeeze(axis=1)
        # print('y_pool', y.shape)

        y = self.mlp(y)
        # print('y_mlp', y.shape)

        y = self.act(y)
        return y
