# Research Roadmap

## 1. Problem

Infrastructure-as-Code (IaC) security analysis tools, even state-of-the-art ones like **GLITCH**, suffer from **high false positive (FP) rates** across technologies such as **Chef, Ansible, and Puppet**.
 This creates **alert fatigue**, where developers ignore or disable tools, limiting their adoption in real-world workflows.

------

## 2. Motivation & Contribution

### Research Question

**Can Large Language Models (LLMs) reduce false positives from static analysis tools without sacrificing detection of real security issues?**

### Key Contribution

We leverage **LLM semantic understanding** to distinguish between:

- **True vulnerabilities vs. False alarms**
- **Real secrets vs. Placeholder examples**
- **Security-critical comments vs. General TODOs**
- **Actual weak crypto usage vs. Documentation mentions**

This directly addresses the **alert fatigue problem**, improving the usability of IaC security analysis.

### Target Security Smells

Our study focuses on categories where **contextual understanding** is essential:

1. **Hard-coded secrets** – separate real secrets from placeholders
2. **Suspicious comments** – detect security-relevant vs. general comments
3. **Weak cryptography** – distinguish actual usage vs. documentation mentions
4. **Insecure HTTP** – identify real communication issues vs. harmless references

------

## 3. Stage 1: Initial Exploration

### Two-Stage Detection Pipeline

1. **Static Analysis (GLITCH)**: Comprehensive detection with high recall but low precision
2. **LLM Post-Filter**: Intelligent filtering of static analysis results

### Evaluation of Approaches

- **Pure LLM Detection**
  - Pros: Context-aware
  - Cons: Inconsistent recall, variable across smell categories
- **Post-Filter LLM (Selected)**
  - Pros: Major precision gains, strong recall retention
  - Outcome: Demonstrated the best trade-off between precision and recall

------

## 4. Stage 2: Model Training Approaches

### Dataset Setup

- **Training/Validation**: Custom-built dataset from IaC samples
- **Test Set**: **Oracle dataset from GLITCH paper**, ensuring fair evaluation
- **Strategy**: **Only combined training** — one model trained jointly on Chef, Ansible, and Puppet, then tested separately on each technology’s test set

### Encoder Approach (CodeBERT / CodeT5 / CodeT5+)

- Input: Raw code snippets with static analysis output
- Task: Binary classification (TP vs. FP)
- Architecture: Encoder + classification head
- Training: Full fine-tuning
- Exploration: **72 different models** tested across architectures and hyperparameters

### Generative Approach (CodeLLaMA-34B / Qwen2.5-Code)

- Input: Prompt with static analysis context
- Task: Generate “TP” or “FP” label
- Architecture: Causal LM with adapter-based tuning (LoRA)
- Observation: Performance **lagged behind encoder models**, less consistent across categories

------

## 5. Stage 3: Evaluation & Comparison

### Metrics

- Precision, Recall, F1-score
- False positive reduction rate
- Cost-benefit analysis (fine-tuned local models vs. API-based solutions)

### Findings

- **Encoder models consistently outperform generative ones** in FP reduction and overall classification accuracy
- Generative LLMs struggled with consistency despite large model sizes (e.g., CodeLLaMA-34B, Qwen2.5-Code)
- Best-performing encoder configurations achieved strong precision/recall balance across all 3 test sets

### Research Questions

1. **Performance**: Do encoder-based classifiers provide more reliable FP filtering than generative reasoning?
   - Preliminary answer: **Yes** — encoders significantly outperform generative approaches
2. **Efficiency**: What is the optimal encoder configuration for deployment?
   - Ongoing exploration via 72-model comparison

------

## 6. Expected Outcomes

1. **Demonstrated FP reduction**: LLM-based filtering boosts precision without degrading recall
2. **Open-source alternatives**: Fine-tuned encoder models rival closed APIs at lower cost
3. **Generative baseline**: Larger models (CodeLLaMA, Qwen2.5) underperform compared to tuned encoders
4. **Practical deployment**: Packaged GLITCH + LLM pipeline with API support
5. **Integration**: CI/CD-ready tools for real-world developer workflows