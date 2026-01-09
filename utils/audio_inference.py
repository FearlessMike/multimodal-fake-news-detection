import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load pretrained Wav2Vec2 (CPU)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

TARGET_SR = 16000  # required by Wav2Vec2

def predict_audio(audio_path: str) -> float:
    """
    Returns an audio credibility / suspicion score in [0,1].
    Higher = more suspicious / less natural.
    """

    # Load audio
def predict_audio(audio_path: str) -> float:
    """
    Returns an audio credibility / suspicion score in [0,1].
    Higher = more suspicious / less natural.
    """

    # Load audio using librosa (Windows-safe)
    waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    inputs = processor(
        waveform,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state.squeeze(0)

    variance = torch.var(hidden_states, dim=0).mean()

    score = torch.sigmoid(variance).item()

    return score


    # Prepare input
    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract hidden representations
    hidden_states = outputs.last_hidden_state.squeeze(0)

    # Compute variance-based uncertainty score
    variance = torch.var(hidden_states, dim=0).mean()

    # Normalize variance to [0,1]
    score = torch.sigmoid(variance).item()

    return score
