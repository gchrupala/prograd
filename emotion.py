from transformers import pipeline
from captum import Saliency
from torchaudio.transforms import Spectrogram



processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
model1 = AutoModelForAudioClassification.from_pretrained("akhmedsakip/wav2vec2-base-ravdess")
model2 = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
model3 = AutoModelForAudioClassification.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model4 = AutoModelForAudioClassification.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")

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
    model1 = AutoModelForAudioClassification.from_pretrained("akhmedsakip/wav2vec2-base-ravdess")
    model2 = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    model3 = AutoModelForAudioClassification.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
    model4 = AutoModelForAudioClassification.from_pretrained("harshit345/xlsr-wav2vec-speech-emotion-recognition")
    
    net1 = Saliency(lambda w: model1(w).logits)
    net2 = Saliency(lambda w: model2(w).logits)
    net3 = Saliency(lambda w: model3(w).logits)
    net4 = Saliency(lambda w: model4(w).logits)
    pitch = Saliency(lambda w: Pitch()(w).reshape(1,1))
    result = []
    for path in paths:
        waveform = waveform, sample_rate = torchaudio.load(path, normalize=True)
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000).mean(dim=0, keepdims=True)
        waveform.require_grad=True
        at1 = net1.attribute(waveform, target=0, abs=False)
        at2 = net2.attribute(waveform, target=0, abs=False)
        at3 = net3.attribute(waveform, target=0, abs=False)
        at4 = net4.attribute(waveform, target=0, abs=False)
        result.append(torch.tensor([corr(at1, at2), corr(at1, at3), corr(at2, at3), corr(at1, at4), corr(at2, at4), corr(at3, at4)]))
    return torch.stack(result)
    
    
