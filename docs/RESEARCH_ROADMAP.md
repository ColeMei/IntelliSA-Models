# Research Roadmap

## 1. Problem

Infrastructure-as-Code (IaC) security analysis tools, even state-of-the-art ones like **GLITCH**, suffer from **high false positive (FP) rates** across technologies such as **Chef, Ansible, and Puppet**. This creates **alert fatigue**, where developers ignore or disable tools, limiting adoption in real workflows.

---

## 2. Motivation & Contribution

### Research Question

**Can Large Language Models (LLMs) reduce false positives from static analysis tools without sacrificing detection of real security issues?**

### Key Contribution

Leverage **LLM semantic understanding** to distinguish between:

- **True vulnerabilities vs. False alarms**
- **Real secrets vs. Placeholder examples**
- **Security‑critical comments vs. General TODOs**
- **Actual weak crypto usage vs. Documentation mentions**

### Target Security Smells

1. **Hard-coded secrets** – separate real secrets from placeholders
2. **Suspicious comments** – detect security‑relevant vs. general comments
3. **Weak cryptography** – distinguish actual usage vs. documentation mentions
4. **Insecure HTTP** – identify real communication issues vs. harmless references

---

## 3. Stage 1: Approach Exploration (Completed)

### Two‑Stage Detection Pipeline

1. **Static Analysis (GLITCH)** for high‑recall detection
2. **LLM Post‑Filter** to prune false positives while retaining true positives

### Approach Selection

- **Post‑Filter LLM** favored over **Pure LLM** for better precision–recall trade‑off and operational stability.
- **Pseudo‑labeling**: Use the post‑filter’s outputs to expand supervised data for downstream training.

---

## 4. Stage 2: Model Training (Completed)

### Dataset Strategy

- **Combined training**: Train a single model jointly on Chef/Ansible/Puppet; evaluate per technology.
- **Label source**: GLITCH detections with **LLM post‑filter pseudo‑labels**.

### Modeling Tracks

- **Encoder classifiers** (e.g., CodeBERT / CodeT5 / CodeT5+): binary TP/FP classification on code + signals.
- **Generative models** (e.g., Code-centric LLMs): prompted TP/FP decision generation (explored in Stage 1, but deprioritized after consistently underperforming encoders on precision/F1).

### Current Direction

- Encoder models are the primary path; **identify and lock an optimal encoder configuration** for reliability and efficiency.

---

## 5. Stage 3: Evaluation & Comparison (Ongoing)

### Metrics

- Precision, Recall, F1‑score
- **False Positive Reduction**
- Cost/latency (local fine‑tuned models vs. API usage)

### Protocol

- Evaluate on **combined** and **per‑technology** test splits.
- **Per‑smell and per‑tech threshold calibration** to balance precision with TP retention.
- Report **aggregate** and **stratified** results.

---

## 6. Expected Outcomes

1. **Demonstrated FP reduction** without degrading recall.
2. **Open model** alternative that rivals closed APIs at lower operating cost.
3. Evidence that **encoders outperform generative baselines** for this task.
4. **Reusable pipeline**: GLITCH → LLM post‑filter → (optional) fine‑tuned encoder.
5. **Integration‑ready artifacts** for practical developer workflows (CLI/API).

---

## 7. Artifact & Deployment (Packaging)

- **CLI + simple API** that consumes GLITCH outputs and returns TP/FP decisions.
- Configurable **decision thresholds** per smell and per technology.
- Lightweight **docs & examples** for CI/CD integration.

---

## 8. Risks & Limitations

- **Label‑quality ceiling** from pseudo‑labels may cap achievable F1.
- **Declarative DSL semantics**: integrity/TLS controls may reside off the target line, challenging purely local reasoning.
- Potential **over‑anchoring** to detector rules if prompts or training signals mirror GLITCH too closely.

---

## 9. Future Work

- **Counterfactual data** to explicitly toggle integrity/SSL and URL↔checksum coupling.
- **Ablations & negative results** (e.g., prompt variants) to map recall–precision trade‑offs.
- Exploration of **context linking** (tying mitigations to the exact URL/source) and limited **control‑flow cues**.

---

## 10. Non‑Goals (for scope clarity)

- Building a new static analyzer from scratch.
- Full program analysis beyond the local IaC change context.
