# Redrob AI Candidate Discovery & Ranking Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Dependencies](https://img.shields.io/badge/Dependencies-Up%20to%20date-brightgreen)
![Constraints](https://img.shields.io/badge/Constraints-5_min_CPU-red)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade, highly optimized machine learning pipeline designed to rank 100,000+ software engineering candidates against complex job descriptions. Built for the **Redrob Intelligent Candidate Discovery Challenge**.

## 📌 Problem Statement & Constraints
The challenge requires sorting a massive pool of nested candidate JSON data based on semantic fit, career history, and behavioral signals. The strict production constraints are:
* **Compute:** CPU-only, max 16GB RAM.
* **Time:** Under 5 minutes total runtime.
* **Network:** Strictly offline (No LLM APIs, no internet access during ranking).

## 🏗️ Architecture

To bypass compute constraints, this engine utilizes a **Two-Stage Split Architecture**. We shift the O(N) embedding complexity to an offline pre-computation stage, allowing the online ranker to operate entirely in vectorized memory space.

```mermaid
graph TD
    subgraph Stage 1: Offline Pre-computation
        A[candidates.jsonl] --> B(JSON Parser & Feature Extractor)
        B --> C[all-MiniLM-L6-v2 Embedder]
        B --> D[Behavioral Heuristics]
        C --> E[(features.parquet)]
        D --> E
    end

    subgraph Stage 2: Online Ranking < 5 mins, CPU only
        F[Job Description text] --> G(Local Embedder)
        E --> H(Vector Engine: np.dot)
        G --> H
        H --> I(Score Fusion Layer)
        I --> J(Risk Detection / Penalties)
        J --> K[Top 100 Selection]
        K --> L[Deterministic Reason Generator]
        L --> M[team_submission.csv]
    end