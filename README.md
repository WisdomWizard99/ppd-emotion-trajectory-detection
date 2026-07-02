# Postpartum Depression Risk Detection via Emotion Trajectory Modeling & Causal Inference

An exploratory NLP pipeline that detects postpartum depression (PPD) risk from anonymized online narratives by modeling **sentence-level emotion trajectories**, applying **interpretable rule-based risk scoring**, and using **causal inference** to test whether positive emotional shifts reduce risk.

> **Note:** This was a designated *high-risk research project* — deliberately scoped to prioritize ambitious, exploratory methods over guaranteed accuracy. The goal was to probe a novel detection approach, map where it breaks, and produce an evidence-backed roadmap for future work.

---

## Motivation

Postpartum depression affects roughly **1 in 7** new parents, yet many cases go undetected. Distress is often expressed in informal, non-clinical language on online forums rather than surfaced through traditional screening. This project asks whether the *dynamics* of emotional language — how tone rises and falls across a post — can serve as an early, scalable signal for PPD risk.

## Approach

The pipeline transforms raw postpartum narratives into risk insights through six stages:

1. **Preprocessing** — Sentence tokenization and stopword removal (NLTK) on free-text posts.
2. **Emotion Trajectory Extraction** — NRCLex maps each sentence to a dominant emotion, producing a sentence-by-sentence emotional timeline.
3. **Trajectory Classification** — Each post is classified into one of four emotional arcs: `neg_to_pos`, `pos_to_neg`, `stable_neg`, `stable_pos`. A turning-point detector flags the largest positive shift in valence as a candidate cognitive-reappraisal moment.
4. **Rule-Based Risk Prediction** — An interpretable model assigns a three-tier risk label (0 = low, 1 = moderate, 2 = high) by combining arc type with crisis-keyword detection.
5. **Intervention Simulation** — Hopeful, future-oriented statements are appended to high-risk posts to measure simulated cognitive-reframing effects, with safeguards that prevent false downscoring when explicit crisis language is present.
6. **Causal Analysis** — A `CausalForestDML` model (econml) estimates the heterogeneous effect of a negative-to-positive emotional shift on high-risk classification, using per-post NRC emotion counts as covariates.

## Key Findings

- The framework flags high-risk posts when sustained negativity co-occurs with crisis keywords, while safeguarding against false downscoring for explicit crisis language.
- Consistent with the high-risk brief, the main value was **mapping the approach's limits**: lexicon-based emotion detection (~55% arc-classification accuracy) struggles with sarcasm, guilt, and mixed emotional tone — strongest on stable arcs (F1 0.69 on `stable_neg`), weakest on transitions.
- Causal analysis identified **sadness, joy, and fear** as the emotional signals most influencing risk change. The average treatment effect was small and not statistically significant, pointing to small, imbalanced data and lexicon limits as the bottlenecks.
- These diagnostics motivate a clear next step: **context-aware transformer models** (e.g., ClinicalBERT, MentalRoBERTa) in place of lexicon-based detection.

## Tech Stack

`Python` · `pandas` · `NLTK` · `NRCLex` · `scikit-learn` · `econml (CausalForestDML)` · `matplotlib` · `seaborn`

## Repository Structure

```
.
├── HRP_code.ipynb        # Full pipeline: preprocessing → emotion arcs → risk → causal analysis
├── data/                 # Anonymized postpartum narrative dataset (not included; see note below)
└── README.md
```

## Getting Started

```bash
# Install dependencies
pip install pandas numpy nltk nrclex scikit-learn econml matplotlib seaborn

# Download NLTK data (run once)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

Then open `HRP_code.ipynb` and run the cells in order.

## Future Work

- Replace lexicon-based emotion detection with context-aware transformer models to better handle sarcasm and nuance.
- Model emotion arcs with sequence models (RNNs / GRUs / Transformers) to capture escalation and recovery.
- Move from rule-based risk to a learned model combining arcs, keywords, and contextual embeddings.
- Validate on larger, more diverse datasets with clinical or crowdsourced labels.
- Integrate human oversight and ethical safeguards (privacy, consent, responsible intervention).

## Ethical Note

This project uses anonymized, publicly shared narratives for research purposes only. It is **not** a diagnostic tool and is not a substitute for professional clinical care. Any real-world application would require human oversight, clinical validation, and careful attention to privacy and consent.

## References

1. De Choudhury, M., Counts, S., & Horvitz, E. (2014). *Characterizing and predicting postpartum changes in emotion and behavior via social media.* SIGCHI Conference on Human Factors in Computing Systems.
2. Amir, S., Coppersmith, G., Carvalho, J. P., Silva, M. J., & Wallace, B. C. (2017). *Quantifying mental health from social media with neural user embeddings.* Machine Learning for Healthcare Conference, PMLR 68.
3. Chancellor, S., & De Choudhury, M. (2020). *Methods in predictive techniques for mental health status on social media: A critical review.* npj Digital Medicine, 3(1), 43.

---

*Author: Deepali Dange · MS in Artificial Intelligence, University of Texas at Austin*
