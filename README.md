# LLM-Based Systematic Literature Review (SLR) Automation 
This repository integrates the OpenAI API to automate the Screening and Data Extraction phases of PRISMA-based Systematic Literature Review (SLR). Specifically the GPT5-Nano model was benchmarked.

# Methodology
To benchmark the AI, we use what we define as "golden datasets". These are datasets that are replications of existing human-conducten SLRs and they are used as ground truth to which the AI's decision are compared to.
### 1. Screening Phase (Title & Abstract / Full-Text)

- The LLM receives:
  - Title  
  - Abstract or Full Text  
  - Inclusion criteria  
  - Exclusion criteria  
- It outputs a **confidence score** between `0.0 – 1.0`  
- The score is converted into a binary decision:
  - `1` = Relevant  
  - `0` = Not Relevant  
- A threshold of **0.5** is used in this study  

### 2. Data Extraction Phase

- The LLM is provided with:
  - Structured prompt instructions  
  - A predefined **JSON schema** of variables  
- It returns extracted information in structured JSON format

# Evaluation & Metrics

Model performance is evaluated by comparing predictions against **golden datasets** using confusion matrix-based metrics.

### Confusion Matrix

- **TP (True Positives)**: Relevant studies correctly identified  
- **FP (False Positives)**: Irrelevant studies incorrectly included  
- **TN (True Negatives)**: Irrelevant studies correctly excluded  
- **FN (False Negatives)**: Relevant studies missed  

### Core Metrics

- **Recall (Sensitivity)**
- **Precision** 
- **Specificity** 
- **Accuracy** 
- **Balanced Accuracy** 
- **F1 Score** 
- **Work Saved (WS)**  
- **WSS@95 (Work Saved over Sampling at Recall R)**  

### Confidence Intervals

- **Wald Confidence Intervals (95%)**  
- **Agresti–Coull Confidence Intervals** (applied for small sample sizes)  

---

# Research Paper
The research paper containing the literature research, research questions, methodology, results and discussion can be found at:

