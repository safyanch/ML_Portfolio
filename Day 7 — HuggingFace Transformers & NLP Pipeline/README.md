---

## Day 7 — HuggingFace Transformers & NLP Pipeline

**Notebook:** `day7_huggingface_intro.ipynb`

### 🎯 Objective
Use pretrained DistilBERT transformer model to classify university
student feedback as positive or negative sentiment — an applied
EdTech AI use case combining NLP with education domain knowledge.

### 🔧 What I Did
- Installed HuggingFace transformers library
- Ran sentiment analysis pipeline in 3 lines of code
- Loaded DistilBERT tokenizer and model manually
- Understood tokenization — input_ids and attention_mask
- Ran manual inference — logits → softmax → prediction
- Classified 20 student feedback sentences from ML/AI course
- Visualized sentiment distribution and confidence scores
- Saved all results to CSV for further analysis

### 📊 Results
| Metric | Value |
|--------|-------|
| Total sentences analyzed | 20 |
| Positive feedback | 12 |
| Negative feedback | 8 |
| Average confidence | 98.2% |
| Most confident prediction | 99.98% |

### 💡 Key Findings
- DistilBERT correctly identified subtle negative feedback such as
  "lectures move too fast" and "too theoretical" with high confidence
- Average confidence of 98.2% shows transformer models generalize
  well to education domain text without any fine-tuning
- Neutral sentences were classified as positive — this is a known
  limitation of binary sentiment models — a multi-class model
  would be needed for production deployment
- This pipeline can scale to hundreds of student responses instantly
  enabling automated feedback monitoring for educators

### 🧠 What I Learned
- HuggingFace pipeline abstracts complex transformer inference
  into 3 lines of code — same model used by real companies
- Tokenization converts words to integer IDs that BERT understands
- [CLS] and [SEP] are special tokens BERT uses for classification
- Softmax converts raw logits to probabilities between 0 and 1
- Pretrained models can be applied to new domains without retraining

### 📈 Visualizations
![Feedback Sentiment Analysis](plots/day7_feedback_sentiment.png)

### 🛠️ Tech Stack
Python · HuggingFace Transformers · DistilBERT · PyTorch · pandas · Matplotlib

### 📂 Output Files
| File | Description |
|------|-------------|
| data/student_feedback_results.csv | All 20 results with labels and confidence scores |
| plots/day7_feedback_sentiment.png | Sentiment distribution and confidence chart |