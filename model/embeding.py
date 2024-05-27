from torch import nn
import numpy


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 16
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)
        return

    def forward(self, img):
        y = self.unfold(img)
        return y.permute(0, 2, 1).contiguous()

