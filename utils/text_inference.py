import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# -----------------------------
# CONFIG
# -----------------------------
MODEL_REPO = "FearlessMike/afriberta-fake-news"
DEVICE = torch.device("cpu")  # Streamlit Cloud uses CPU

# -----------------------------
# LOAD MODEL ONCE (CACHED)
# -----------------------------
_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model

    if _model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        _model.to(DEVICE)
        _model.eval()

    return _tokenizer, _model


# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def predict_text(text: str) -> float:
    """
    Returns probability that the text is FAKE (0.0 – 1.0)
    """
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    # Assumption: label 1 = FAKE (same as training)
    fake_prob = probs[0, 1].item()
    return fake_prob
