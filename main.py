import torch
import torch.nn as nn
import torchaudio.datasets
import torchaudio.functional as A
import torch.nn.functional as F
from captum.attr import Saliency
import logging
#from torchaudio.transforms import Loudness, Spectrogram
from torchaudio.transforms import Spectrogram

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    def forward(self, w):
        return torch.cat([msq(w), w.std(dim=1), w.max(dim=1)[0] + msq(w) + w.std(dim=1)]).unsqueeze(dim=0)

class Pitch(nn.Module):
    def __init__(self):
        super(Pitch, self).__init__()
        self.spec = Spectrogram(n_fft=800)
    def forward(self, w):
        return softargmax(self.spec(w).mean(dim=2), beta=10)
    
def corr(u, v):
    #return (u * v).norm()
    return torch.corrcoef(torch.cat([u, v]))[0,1]

def msq(v):
    return (v*v).mean(dim=1)

def softargmax(input, beta=100):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result   

def main():
    logging.getLogger().setLevel(logging.INFO)
    torch.set_grad_enabled(True)
    dataset = torchaudio.datasets.LibriLightLimited("data/in", subset="10min", download=True)
    net = Saliency(Net())
    msqq = Saliency(lambda w: msq(w).unsqueeze(dim=0))
    std = Saliency(lambda w: w.std(dim=1).unsqueeze(dim=0))
    maxx = Saliency(lambda w: w.max(dim=1)[0].unsqueeze(dim=0))
    #loud = Saliency(lambda w: Loudness(16000)(w).reshape(1,1))
    pitch = Saliency(lambda w: Pitch()(w).reshape(1,1))
    for item in dataset:
        w, *_ = item
        w.requires_grad = True
        gnet0 = net.attribute(w, target=0, abs=False)
        gnet2 = net.attribute(w, target=2, abs=False)
        gmsq = msqq.attribute(w, target=0, abs=False)
        gstd = std.attribute(w, target=0, abs=False)
        gmax = maxx.attribute(w, target=0, abs=False)
        #gloud = loud.attribute(w, target=0, abs=False)
        gpitch = pitch.attribute(w, target=0, abs=False)
        logging.info(f"Corr: net2 vs msq  = {corr(gnet2, gmsq)}")
        logging.info(f"Corr: net2 vs std  = {corr(gnet2, gstd)}")
        logging.info(f"Corr: net2 vs max  = {corr(gnet2, gmax)}")
        #logging.info(f"Corr: net2 vs loud = {corr(gnet2, gloud)}")
        logging.info(f"Corr: net2 vs pitch = {corr(gnet2, gpitch)}")
        print()




main()

