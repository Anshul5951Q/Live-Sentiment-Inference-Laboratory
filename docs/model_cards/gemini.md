# Model Card: Google Gemini (Generative Foundation Model In-Context Classifier)

## 1. Model Details
- **Model Name:** Google Gemini (`gemini-2.5-flash` / `gemini-3.5-flash-lite`)
- **Developer:** Google DeepMind / Google Cloud AI.
- **Provider Role:** Academic Provider Substitution for GPT (due to open free-tier API quotas and developer accessibility).
- **Interface:** REST / HTTP API via the official `google-genai` SDK (`v1beta` endpoint).
- **Paradigm:** Generative Foundation Model accessed through prompt-based in-context learning.
- **Prompting Modes:**
  1. **Zero-Shot Prompting:** Task description, class definitions, strict JSON schema specification, zero exemplary demonstrations.
  2. **Few-Shot Prompting:** System instruction, 6 balanced demonstrations (2 Negative, 2 Neutral, 2 Positive) isolated from the evaluation benchmark.
- **License:** Google AI Terms of Service.

## 2. Intended Use
- **Primary Use:** In-context sentiment classification without parameter fine-tuning, natural language rationale generation, and synthetic review generation.
- **Out-of-Scope Use:** Direct high-throughput batch classification without rate-limit throttling (restricted to free-tier quota limits: 15 RPM, 20 requests/day per model).

## 3. Important Conceptual Distinctions
- **Prompt-Based Inference vs Fine-Tuning:** The weights of the Gemini foundation model are **never updated** during review generation or sentiment classification. The model remains frozen.
- **Confidence vs Calibrated Probability:** Gemini returns a scalar heuristic confidence ($0.0 \le c \le 1.0$), which reflects internal prompt certainty but does **not** constitute a calibrated multiclass probability distribution. Brier scores and ECE are omitted for LLM outputs to maintain metric integrity.

## 4. Strengths & Limitations
- **Strengths:** Produces natural human-readable explanations ("reason") justifying its predictions; handles diverse domain expressions zero-shot.
- **Limitations:** API network latency (~800–2,000 ms per sample), susceptibility to rate limits (HTTP 429), and non-deterministic text generation requiring JSON parsing guards.

