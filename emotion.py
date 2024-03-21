from transformers import pipeline
from captum import Saliency
from torchaudio.transforms import Spectrogram



processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForAudioClassification.from_pretrained("akhmedsakip/wav2vec2-base-ravdess")




class Pitch(nn.Module):
    def __init__(self):
        super(Pitch, self).__init__()
        self.spec = Spectrogram(n_fft=800)
    def forward(self, w):
        return softargmax(self.spec(w).mean(dim=2), beta=10)
    
def corr(u, v):
    #return (u * v).norm()
    return torch.corrcoef(torch.cat([u, v]))[0,1]

def softargmax(input, beta=100):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result

def correlation(paths):
    net = Saliency(lambda w: model(w).logits)
    pitch = Saliency(lambda w: Pitch()(w).reshape(1,1))
    result = []
    for path in paths:
        waveform = waveform, sample_rate = torchaudio.load(path, normalize=True)
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        print(waveform.shape)
        at1 = net.attribute(waveform, target=0, abs=False)
        at2 = pitch.attribute(waveform, target=0, abs=False)
        result.append(corr(at1, at2))
    return result
    
    
