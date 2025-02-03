import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
        )
        self.local = nn.Sequential(
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256,128, kernel_size=3, padding=1)
        )
        self.intermediate = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp1 = nn.Linear(256,128)
        self.mlp2 = nn.Sequential(
            #nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        x = self.E(x)
        local_fea = self.local(x)
        global_fea = self.intermediate(x).squeeze(-1).squeeze(-1)
        global_fea = self.mlp1(global_fea)
        out = self.mlp2(global_fea)

        return local_fea, global_fea, out
