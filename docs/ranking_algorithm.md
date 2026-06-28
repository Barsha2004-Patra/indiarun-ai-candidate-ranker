# Ranking Pipeline & Score Fusion

## 1. Semantic Score (Weight: 55%)
Embeddings generated via `all-MiniLM-L6-v2` are strictly normalized to a length of 1 during the offline stage. 
* **Mathematical Advantage:** Because they are normalized, the dot product `A · B` is mathematically identical to Cosine Similarity. 
* **Compute Advantage:** `np.dot()` is heavily optimized in C/Fortran (BLAS), executing across 100k candidates in ~0.2 seconds.

## 2. Behavioral Score (Weight: 25%)
Real-world recruitment relies on activity signals. The score is a composite of:
* `recruiter_response_rate` (Highest internal weight)
* `profile_completeness`
* `notice_period_days` (Inversely scaled: shorter is better)

## 3. Experience Normalization (Weight: 20%)
Candidate experience is scaled to a `[0, 1]` index based on the JD's maximum ideal tier (configurable in `settings.yaml`, defaulting to 15 years).

## 4. Risk / Honeypot Penalty
A separate business rules engine acts as a multiplier penalty. If `is_risk_honeypot` evaluates to True, the final fused score is multiplied by `0.1`, plummeting the candidate's rank securely without deleting the record.