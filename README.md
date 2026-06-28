# 🧠 India Run Hack2Skill – AI Candidate Discovery & Ranking System

An intelligent AI-powered candidate discovery and ranking system developed for the **India Run Hack2Skill – Data & AI Challenge**.

This project implements a scalable semantic ranking pipeline that goes beyond traditional keyword matching by combining semantic similarity, behavioral intelligence, and risk analysis to identify the most relevant candidates for a given job description.

---

# 🚀 Problem Statement

Develop a Proof of Concept that intelligently ranks candidates instead of simply filtering them.

The system should:

* Understand complex job descriptions
* Perform semantic matching beyond keywords
* Incorporate candidate profile information and behavioral signals
* Produce a fast, accurate, explainable ranked shortlist

---

# ✨ Features

* Semantic candidate ranking
* Context-aware job description understanding
* Behavioral signal scoring
* Risk detection and penalty scoring
* Explainable ranking reasons
* Modular architecture
* Efficient offline feature generation
* Scalable preprocessing pipeline
* Validated submission CSV generation

---

# 🏗️ Project Structure

```
indiarun-ai-candidate-ranker/

├── configs/
│   └── settings.yaml
│
├── data/
│   ├── job_description.txt
│   └── team_submission.csv
│
├── src/
│   ├── data_pipeline/
│   │   └── offline_processor.py
│   │
│   ├── ranking/
│   │   └── ranker.py
│   │
│   ├── reasoning/
│   │   └── generator.py
│   │
│   ├── scoring/
│   │   ├── semantic.py
│   │   ├── behavioral.py
│   │   └── risk.py
│   │
│   ├── config.py
│   ├── dataloader.py
│   └── main.py
│
├── validate_submission.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ Ranking Pipeline

```
Job Description
        │
        ▼
Semantic Understanding
        │
        ▼
Candidate Feature Loading
        │
        ▼
Semantic Matching
        │
        ▼
Behavioral Scoring
        │
        ▼
Risk Assessment
        │
        ▼
Weighted Score Fusion
        │
        ▼
Candidate Ranking
        │
        ▼
Reason Generation
        │
        ▼
Submission CSV
```

---

# 🧠 Core Components

## Semantic Scoring

Evaluates contextual similarity between candidate profiles and the job description using semantic representations rather than simple keyword matching.

---

## Behavioral Intelligence

Incorporates behavioral indicators such as candidate activity and engagement signals into the ranking process.

---

## Risk Analysis

Identifies potentially unreliable candidate profiles and applies configurable penalties to improve ranking quality.

---

## Explainability

Generates human-readable reasoning describing why each candidate received their ranking.

---

# ⚡ Performance

Designed with scalability in mind through:

* Offline preprocessing
* Efficient feature loading
* Modular scoring pipeline
* Optimized ranking workflow
* Lightweight inference

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/Barsha2004-Patra/indiarun-ai-candidate-ranker.git
cd indiarun-ai-candidate-ranker
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Execute the ranking pipeline:

```bash
python src/main.py
```

Validate the generated submission:

```bash
python validate_submission.py
```

---

# 📦 Output

The system generates a ranked submission CSV containing:

* Candidate ID
* Final Rank
* Ranking Score
* Generated Reason (if enabled)

---

# 📈 Technology Stack

* Python
* Pandas
* NumPy
* PyYAML
* Parquet
* Machine Learning-based Semantic Ranking
* Modular Scoring Pipeline

---

# 📊 Design Highlights

* Semantic-first ranking
* Modular architecture
* Configurable scoring weights
* Explainable outputs
* Fast offline inference
* Easy extensibility

---

# 🔒 Notes

Large datasets and generated artifacts are intentionally excluded from this repository through `.gitignore`.

This keeps the repository lightweight while ensuring reproducibility.

---

# 📄 License

This project is released under the MIT License.

---

# 👩‍💻 Author

**Barsha Patra**

GitHub: https://github.com/Barsha2004-Patra

---

# 🙏 Acknowledgements

Developed as part of the **India Run Hack2Skill – Data & AI Challenge**.

Special thanks to the organizers for providing the problem statement and evaluation framework.
