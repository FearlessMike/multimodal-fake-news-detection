import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import numpy as np

# ======================================================
# CONFIG
# ======================================================
DEVICE = torch.device("cpu")

AFRIBERTA_REPO = "FearlessMike/afriberta-fake-news"
XLMR_REPO = "FearlessMike/xlm-roberta-fake-news"

LOG_REG_PATH = "model/log_reg_model.pkl"
TFIDF_PATH = "model/tfidf_vectorizer.pkl"

# Ensemble weights (must sum to 1.0)
W_AFRIBERTA = 0.4
W_XLMR = 0.4
W_LOGREG = 0.2

# ======================================================
# LOADERS (CACHED)
# ======================================================
_af_tokenizer = None
_af_model = None

_xlmr_tokenizer = None
_xlmr_model = None

_logreg = None
_tfidf = None


def load_afriberta():
    global _af_tokenizer, _af_model
    if _af_model is None:
        _af_tokenizer = AutoTokenizer.from_pretrained(AFRIBERTA_REPO)
        _af_model = AutoModelForSequenceClassification.from_pretrained(AFRIBERTA_REPO)
        _af_model.to(DEVICE).eval()
    return _af_tokenizer, _af_model


def load_xlmr():
    global _xlmr_tokenizer, _xlmr_model

    if _xlmr_model is None or _xlmr_tokenizer is None:
        try:
            _xlmr_tokenizer = AutoTokenizer.from_pretrained(XLMR_REPO)
            _xlmr_model = AutoModelForSequenceClassification.from_pretrained(XLMR_REPO)
            _xlmr_model.to(DEVICE)
            _xlmr_model.eval()
        except Exception as e:
            print("XLM-RoBERTa loading failed:", e)
            return None, None

    return _xlmr_tokenizer, _xlmr_model



def load_logreg():
    global _logreg, _tfidf
    if _logreg is None or _tfidf is None:
        with open(LOG_REG_PATH, "rb") as f:
            _logreg = pickle.load(f)
        with open(TFIDF_PATH, "rb") as f:
            _tfidf = pickle.load(f)
    return _logreg, _tfidf


# ======================================================
# MODEL PREDICTORS
# ======================================================
def _predict_transformer(text, tokenizer, model):
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

    # label 1 = FAKE
    return probs[0, 1].item()


def _predict_logreg(text, model, vectorizer):
    X = vectorizer.transform([text])
    return model.predict_proba(X)[0, 1]


# ======================================================
# ENSEMBLE INFERENCE
# ======================================================
def predict_text(text: str) -> float:
    """
    Returns ensemble probability that the text is FAKE (0.0 – 1.0)
    Uses AfriBERTa + XLM-RoBERTa + Logistic Regression
    """

    # Load models
    af_tok, af_model = load_afriberta()
    xl_tok, xl_model = load_xlmr()
    logreg, tfidf = load_logreg()

    # AfriBERTa prediction
    af_score = _predict_transformer(text, af_tok, af_model)

    # XLM-R prediction (fallback-safe)
    if xl_tok is not None and xl_model is not None:
        xlmr_score = _predict_transformer(text, xl_tok, xl_model)
    else:
        xlmr_score = af_score  # fallback if XLM-R fails

    # Logistic Regression prediction
    logreg_score = _predict_logreg(text, logreg, tfidf)

    # Weighted ensemble
    final_score = (
        W_AFRIBERTA * af_score +
        W_XLMR * xlmr_score +
        W_LOGREG * logreg_score
    )

    return float(final_score)
