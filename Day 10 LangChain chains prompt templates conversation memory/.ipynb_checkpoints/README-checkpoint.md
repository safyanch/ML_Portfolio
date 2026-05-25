## Day 10 — LangChain Chains, Templates & Memory

**Notebook:** `day10_langchain_basics.ipynb`

### 🎯 Objective
Build reusable AI chains using LangChain connected to Groq/Llama3.
Create 3 practical EdTech tools and an ML tutor with memory.

### 🔧 What I Built
- First LangChain chain — PromptTemplate | LLM | OutputParser
- ML concept explainer — explains any topic with analogy
- Quiz generator — generates 5 MCQs on any ML topic automatically
- Student feedback summarizer — produces structured analysis report
- ML tutor with ConversationBufferMemory — remembers student name
  and all previous questions across the session

### 💡 Key Findings
- LangChain reduces API call code from 8 lines to 3 lines
- Prompt templates make AI tools reusable across topics
- ConversationBufferMemory passes full history to LLM every turn
- Quiz generator directly applicable to university teaching workflow

### 🛠️ Tech Stack
Python · LangChain · Groq API · Llama3 · pandas

### 📂 Output Files
| File | Description |
|------|-------------|
| data/day10_conversation.txt | Full ML tutor conversation log |
| data/day10_conversation.csv | Conversation as structured data |
| data/day10_feedback_report.txt | Student feedback analysis report |