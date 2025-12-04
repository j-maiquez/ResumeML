# ResumeML: Bias Detection in Resume Screening Systems

## Overview

**ResumeML** is a comprehensive bias detection and fairness audit tool for resume screening systems. It analyzes how different applicant characteristics (location, education prestige, technical skills, experience level) influence resume-to-job-description matching scores. Using embedding-based similarity analysis combined with explainable machine learning (SHAP), ResumeML identifies potential biases in automated resume screening and helps organizations build fairer hiring processes.

## Project Goals

- **Detect Bias**: Identify if resume screening systems systematically disadvantage candidates based on protected or proxy characteristics
- **Understand Mechanisms**: Use SHAP explainability to understand which features drive matching scores
- **Audit Fairness**: Provide actionable insights on whether bias proxy features (location, school tier, years of experience) are changing screening outcomes

## Who Should Use This?

- **Researchers** studying algorithmic fairness in hiring
- **Data Scientists** building or auditing recruitment systems
- **HR Professionals** ensuring fair hiring practices
- **Companies** committed to reducing bias in talent acquisition

---

## Dataset

This project uses the [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data) from Kaggle, which contains:
- **Resume categories**: Information Technology, Engineering, and others
- **Resume text**: Raw resume text
- **Resume HTML**: Scraped HTML content

For this analysis, we focus on **Information Technology** and **Engineering** categories (~238 resumes).

---

## Pipeline Overview

The analysis follows a complete end-to-end workflow:

```
Raw Resumes → Skill Extraction → Embeddings → Similarity Scoring → Feature Engineering → Model Training → SHAP Interpretation
```

### 1. **Skill Extraction**
- Uses **Google Gemini API** to extract professional skills from resumes
- Returns structured JSON with up to 30 most relevant skills per resume
- Includes both hard skills (Python, AWS, etc.) and soft skills (Leadership, Communication)

### 2. **Embedding Generation**
- Converts resume text into dense vector representations using **Ollama** local embeddings
- Model: `dengcao/Qwen3-Embedding-0.6B:Q8_0`
- Batch processing for efficiency (batch size: 100)
- Robust error handling for failed embeddings

### 3. **Similarity Scoring**
- Computes cosine similarity between job description embedding and each resume embedding
- Scores range from 0 (completely dissimilar) to 1 (perfect match)
- Used as the target variable for fairness analysis

### 4. **Feature Engineering**
Extracts interpretable features from resume text to detect bias proxies:

| Feature Category       | Features                                                      | Purpose                      |
|------------------------|---------------------------------------------------------------|------------------------------|
| **Controls**           | `matched_skills_count`, `skills_coverage`, `years_experience` | Productivity/quality metrics |
| **Education Prestige** | `school_tier`                                                 | Detect prestige bias         |
| **Location**           | `is_us_location`, state one-hot encoding                      | Detect geographic bias       |

### 5. **Model Training** (Ridge, Decision Tree, Random Forest)
- **Ridge Regression**: Linear baseline
- **Decision Tree**: Interpretable non-linear model
- **Random Forest**: Best-performing model with feature importances
- GridSearchCV used for hyperparameter tuning
- Train/test split: 80/20

### 6. **SHAP Interpretation**
- **KernelExplainer** provides instance-level explanations
- Summary plots show which features most influence similarity scores
- Identifies if bias proxy features are changing outcomes
- Detects systematic unfairness patterns

---

## Installation & Setup

### Requirements

```bash
# Core dependencies
python>=3.9
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
shap>=0.41.0
matplotlib>=3.4.0

# API & Embedding dependencies
google-generativeai>=0.3.0  # Gemini API
ollama>=0.1.0              # Local embeddings
nest_asyncio>=1.5.0        # Async support
tqdm>=4.62.0               # Progress bars

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/thubpham/resume-scanner.git
cd resume-scanner/ResumeML
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up Gemini API**:
   - Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create `.env` file:
   ```
   GEMINI_API=your_api_key_here
   ```

5. **Set up Ollama** (for local embeddings):
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0
   ollama serve  # Start Ollama server in separate terminal
   ```

---

## Quick Start

### Running the Full Analysis

```python
# 1. Load and filter dataset
import pandas as pd
from resume_scanner import extract_skills_google, embed_resumes, calculate_cosine_similarity

filtered_df = pd.read_csv("final_df_w_all_features.csv")
filtered_df = filtered_df[
    (filtered_df["Category"] == "INFORMATION-TECHNOLOGY") | 
    (filtered_df["Category"] == "ENGINEERING")
]

# 2. Extract skills using Gemini
result = await extract_skills_google(filtered_df)
filtered_df["skills_v2"] = result

# 3. Generate embeddings
df_with_embeddings = await embed_resumes(
    df=filtered_df,
    text_column='Resume_str',
    embedding_column='embeddings',
    model='dengcao/Qwen3-Embedding-0.6B:Q8_0'
)

# 4. Score resumes against job description
job_embedding = await get_embeddings("Your job description here")
df_scores = calculate_cosine_similarity(df_with_embeddings, job_embedding)

# 5. Extract features for bias analysis
df_features = pd.DataFrame([extract_comprehensive_features(t) for t in df['Resume_str']])

# 6. Train SHAP interpretable model
import shap
from sklearn.ensemble import RandomForestRegressor

X = df_features.drop(columns=['us_state', 'token_count'])
y = df_scores['scores']

rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

## File Structure

```
ResumeML/
├── resume_scanner.ipynb          # Main analysis notebook
├── requirements.txt              # Python dependencies
├── .env                          # API keys (create this)
├── data/
│   ├── final_df_w_all_features.csv
│   ├── resume_embeddings.csv
│   └── resume_embeddings_with_scores.csv
├── models/
│   ├── ridge_model.pkl
│   ├── rf_v1.pkl
│   └── decision_tree.pkl
└── outputs/
    ├── shap_summary_plot.png
    ├── shap_bar_plot.png
    └── fairness_report.json
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ollama.ResponseError` | Ensure Ollama server is running: `ollama serve` |
| `google.api_core.exceptions.NotFound` | Check Gemini API key in `.env` |
| `SHAP KernelExplainer timeout` | Reduce background sample size or use `TreeExplainer` instead |
| Memory errors during embedding | Reduce batch size or split dataset |

---

## Contributing

We welcome contributions! Areas for improvement:
- [ ] Support for additional embedding models
- [ ] Interactive Fairness Dashboard
- [ ] Additional bias detection metrics (disparate impact, demographic parity)
- [ ] Multi-language support

---

## Citation

If you use ResumeML in your research, please cite:

```bibtex
@software{ResumeML,
  author = {Pham, Bao Thu},
  title = {ResumeML: Bias Detection in Resume Screening Systems},
  year = {2025},
  url = {https://github.com/thubpham/resume-scanner}
}
```

---

## License

MIT

---

## Acknowledgments

- Resume dataset from [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data)
- SHAP library for interpretability: [SHAP GitHub](https://github.com/slundberg/shap)
- Ollama for local embeddings: [Ollama](https://ollama.ai)
- Google Gemini API for skill extraction: [Google AI](https://ai.google.dev/)
