import torch
import torch.nn as nn
import torchaudio.datasets
import torchaudio.functional as A
import torch.nn.functional as F
from captum.attr import Saliency
import logging

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    def forward(self, w):
        return torch.cat([msq(w), w.std(dim=1), 0.1 * msq(w) + 0.9 * w.std(dim=1)]).unsqueeze(dim=0)

    
def corr(u, v):
    #return (u * v).norm()
    return torch.corrcoef(torch.cat([u, v]))[0,1]

def msq(v):
    return (v*v).mean(dim=1)

def main():
    logging.getLogger().setLevel(logging.INFO)
    dataset = torchaudio.datasets.LibriLightLimited("data/in", subset="10min", download=True)
    net = Saliency(Net())
    msqq = Saliency(lambda w: msq(w).unsqueeze(dim=0))
    std = Saliency(lambda w: w.std(dim=1).unsqueeze(dim=0))
    maxx = Saliency(lambda w: w.max(dim=1)[0].unsqueeze(dim=0))
    for item in dataset:
        w, *_ = item
        w.requires_grad = True
        gnet0 = net.attribute(w, target=0, abs=False)
        gnet2 = net.attribute(w, target=2, abs=False)
        gmsq = msqq.attribute(w, target=0, abs=False)
        gstd = std.attribute(w, target=0, abs=False)
        gmax = maxx.attribute(w, target=0, abs=False)
        logging.info(f"Corr: net2 vs msq= {corr(gnet2, gmsq)}")
        logging.info(f"Corr: net2 vs std = {corr(gnet2, gstd)}")
        logging.info(f"Corr: net2 vs max = {corr(gnet2, gmax)}")
        print()




main()

