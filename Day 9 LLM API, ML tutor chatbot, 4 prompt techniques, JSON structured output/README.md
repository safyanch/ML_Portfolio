## Day 9 — LLM APIs & Prompt Engineering

**Notebook:** `day9_llm_api_prompt_engineering.ipynb`

### 🎯 Objective
Call large language models from Python, build an ML tutor
chatbot with conversation memory, and master 4 prompt
engineering techniques for production AI development.

### 🔧 What I Did
- Connected to LLM API (Groq/Llama3) from Python
- Built ML tutor chatbot with full conversation history
- Implemented system prompts for domain-specific behavior
- Tested 4 prompt engineering techniques and compared results
- Built structured output extractor using JSON mode prompting
- Extracted sentiment, score, and action items from student feedback

### 📊 Prompt Techniques Compared
| Technique | Best For | Reliability |
|-----------|----------|-------------|
| Zero-shot | Simple tasks | Medium |
| Few-shot | Classification | High |
| Chain-of-thought | Complex reasoning | High |
| JSON mode | Structured output | Very High |

### 💡 Key Findings
- Few-shot prompting improved classification consistency vs zero-shot
- Chain-of-thought best for explaining complex ML concepts step by step
- JSON mode enables LLM output to be used directly in data pipelines
- System prompts dramatically shape model personality and response style

### 🛠️ Tech Stack
Python · Groq API · Llama 3 · pandas · python-dotenv

### 📂 Output
| File | Description |
|------|-------------|
| data/day9_structured_feedback.csv | Structured feedback analysis results |