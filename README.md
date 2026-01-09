[🤗 AfriBERTa Fake News Model](https://huggingface.co/FearlessMike/afriberta-fake-news)

## 🔗 Pretrained Models (Hosted on Hugging Face)

This project uses **fine-tuned transformer models hosted on Hugging Face Hub**.
The models are **not stored in this GitHub repository** to keep the repo lightweight
and deployment-friendly.

### Text Models
- **AfriBERTa (Fake News Detection)**  
  🔗 https://huggingface.co/FearlessMike/afriberta-fake-news

- **XLM-RoBERTa (Fake News Detection)**  
  🔗 https://huggingface.co/FearlessMike/xlm-roberta-fake-news *(to be uploaded)*

### Loading Strategy
Models are automatically downloaded at runtime using the Hugging Face `transformers`
library:

```python
AutoTokenizer.from_pretrained("FearlessMike/afriberta-fake-news")
AutoModelForSequenceClassification.from_pretrained("FearlessMike/afriberta-fake-news")
