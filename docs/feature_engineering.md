# Feature Engineering Strategy

Our pipeline converts nested JSON structures into flat, numerical heuristics.

## Derived Features
1. **`avg_tenure_months`**: Calculated by summing all `duration_months` in the `career_history` array and dividing by the array length. Identifies job hoppers.
2. **`expert_skill_count`**: Filters the `skills` array for proficiency `== 'expert'`. Used for honeypot detection.
3. **`is_risk_honeypot`**: A strict boolean boolean flag activated when a candidate claims >5 expert skills but possesses <1 year of real-world experience.

## Text Aggregation Strategy
To create the optimal context window for the embedding model, we do not embed the entire JSON dump. We construct a targeted semantic block:
`Title: {current_title}. Summary: {summary}. Experience: {Top 3 Career History Descriptions}`