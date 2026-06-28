# Deterministic Reason Generation

The Hackathon specification requires a 1-2 sentence justification for the top 100 candidates. 

## Design Choice: No LLMs
Using GPT-4, Claude, or a local LLM API violates the strict offline, no-network constraint. Local generative models (like LLaMA-3 8B) would exceed the 16GB RAM constraint and the 5-minute time limit.

## Implementation: Decision Trees
We utilize a dynamic, rules-based string generator (`src/reasoning/generator.py`). 
It evaluates the candidate's precise floating-point scores and constructs a highly specific, factual string based on explicit profile features.

**Example Output:**
> "Strong semantic alignment with core JD requirements (0.88 score). Brings 7.5 years of experience as a Senior AI Engineer. Highly engaged candidate with excellent recruiter response rates."

* **Advantage:** 100% immune to hallucination. Guaranteed to accurately reflect the internal scoring weights.