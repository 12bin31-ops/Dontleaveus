# 🔮 Customer Churn Prediction with Harness Engineering

Advanced machine learning system for predicting customer churn with automated reporting, LLM-based insights using Harness Engineering, and complete pipeline automation.

## 🌟 Key Features

### 1. **Machine Learning Pipeline**
- Multiple ML algorithms comparison (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning with Optuna
- Automated model selection based on F1 score
- Cross-validation and stratified splitting

### 2. **Explainability & Insights**
- **SHAP (SHapley Additive exPlanations)** for model interpretability
- **Feature importance** extraction
- **LLM-based insight generation** using advanced prompting techniques:
  - **Chain of Thought (CoT)**: Step-by-step reasoning
  - **Self-Consistency**: Multiple perspectives synthesis
  - **ReAct**: Reason-Act-Observe cycle
  - **Memory Compaction**: Concise key findings

### 3. **Automated Reporting**
- HTML/PDF reports with rich visualizations
- Confusion matrix, ROC curve, feature importance charts
- Automated high-risk customer identification
- Executive summary generation

### 4. **Pipeline Automation**
- **APScheduler** for periodic execution
- Scheduled model retraining
- Automated inference on new data
- Report generation on schedule
- Production-ready error handling

### 5. **Data Handling**
- Automatic preprocessing and feature engineering
- Missing value handling
- Categorical encoding
- Feature scaling
- SMOTE for imbalanced data

## 📋 Project Structure

```
customer-churn-prediction/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                      # Raw data
│   └── processed/                # Processed data
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Utility functions
│   ├── data_loader.py            # Data loading & preprocessing
│   ├── model_trainer.py          # Model training & tuning
│   ├── model_evaluator.py        # Model evaluation & SHAP
│   ├── insight_generator.py      # LLM insights (Harness Engineering)
│   └── report_generator.py       # Automated report generation
├── pipeline/
│   ├── __init__.py
│   ├── main.py                   # Main pipeline orchestration
│   └── scheduler.py              # APScheduler configuration
├── models/                        # Trained models
├── reports/                       # Generated reports
├── templates/
│   └── report_template.html       # HTML report template
├── tests/                         # Unit tests
├── scripts/
│   ├── train_model.py            # Training script
│   ├── run_inference.py          # Inference script
│   └── generate_report.py        # Report generation script
├── requirements.txt              # Dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose setup
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Docker (optional)

### Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
export ANTHROPIC_API_KEY="your-api-key"  # For LLM insights
```

### Usage

#### Option 1: Direct Python Execution

```bash
# Run training pipeline
python pipeline/main.py

# Run inference on new data
python scripts/run_inference.py --data-path data/raw/new_customers.csv
```

#### Option 2: Docker

```bash
# Build image
docker build -t customer-churn-prediction .

# Run container
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/reports:/app/reports \
           customer-churn-prediction
```

#### Option 3: Docker Compose

```bash
docker-compose up
```

## 📊 Pipeline Workflow

### Training Pipeline
```
1. Load Data
   ↓
2. Exploratory Data Analysis
   ↓
3. Preprocessing (scaling, encoding, SMOTE)
   ↓
4. Train/Test Split
   ↓
5. Train Multiple Models
   ↓
6. Hyperparameter Tuning (Optuna)
   ↓
7. Model Evaluation (metrics, SHAP)
   ↓
8. Generate LLM Insights (Harness Engineering)
   ↓
9. Create Visualizations
   ↓
10. Generate Automated Report (HTML/PDF)
   ↓
11. Save Best Model
```

### Inference Pipeline
```
1. Load New Data
   ↓
2. Preprocess (same as training)
   ↓
3. Load Trained Model
   ↓
4. Generate Predictions
   ↓
5. Identify High-Risk Customers
   ↓
6. Generate Alert/Report
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
models:
  classifiers:
    - name: "XGBoost"
      params:
        n_estimators: 100
        max_depth: 6

hyperparameter_tuning:
  enabled: true
  n_trials: 50

scheduling:
  enabled: true
  retrain:
    day_of_week: "mon"  # Monday
    hour: 9
    minute: 0
```

## 🤖 Harness Engineering Techniques

### 1. Chain of Thought (CoT)
Encourages step-by-step reasoning for systematic analysis:
```python
insight = insight_generator.chain_of_thought(
    context="Churn metrics and features",
    question="What are key churn drivers?"
)
```

### 2. Self-Consistency
Generates multiple perspectives and synthesizes them:
```python
insight = insight_generator.self_consistency(
    context="Churn data",
    question="What business actions to take?"
)
```

### 3. ReAct (Reason-Act-Observe)
Iterative cycle of reasoning, acting, and observing:
```python
insight = insight_generator.react_reasoning(
    context="Churn analysis",
    question="How to prevent churn?"
)
```

### 4. Memory Compaction
Compresses findings into concise key takeaways:
```python
takeaways = insight_generator.memory_compaction(findings)
```

## 📈 Model Comparison

The pipeline trains and compares:

| Model | Pros | Cons |
|-------|------|------|
| Logistic Regression | Interpretable, fast | Lower accuracy |
| Random Forest | Good balance | Slower training |
| XGBoost | High accuracy | Complex tuning |
| LightGBM | Fast, memory efficient | Prone to overfitting |

## 🔍 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy  
- **Recall**: True positive detection rate
- **F1 Score**: Harmonic mean (primary metric)
- **ROC-AUC**: Classification performance

## 📝 Example Report Output

The pipeline generates comprehensive reports with:

1. **Executive Summary**
2. **Model Performance Metrics** (cards with color coding)
3. **Top Features Driving Churn** (table)
4. **High-Risk Customers** (top 10 with risk levels)
5. **AI-Generated Insights** (from Harness Engineering)
6. **Recommended Actions** (prioritized)
7. **Visualizations**:
   - Feature Importance Chart
   - Confusion Matrix
   - ROC Curve

## 🔄 Automated Scheduling

With APScheduler, the pipeline can run on schedule:

```yaml
scheduling:
  retrain:           # Weekly Monday 9 AM
  inference:         # Daily 8 AM
  report_generation: # Weekly Tuesday 10 AM
```

Configure in `config/config.yaml` and let it run 24/7.

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📦 Deployment

### Cloud Deployment (AWS Example)

```bash
# Create ECR repository
aws ecr create-repository --repository-name customer-churn-prediction

# Build and push
docker build -t customer-churn-prediction .
docker tag customer-churn-prediction:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/customer-churn-prediction:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/customer-churn-prediction:latest

# Deploy to ECS/EKS
```

## 🔐 Security Considerations

- Use environment variables for API keys
- Never commit credentials to repo
- Implement access control for reports
- Log all pipeline executions
- Use HTTPS for API endpoints

## 📊 Dataset

The project is tested with:
- **Kaggle Telco Customer Churn** (~7K samples)
- Or any similar structured dataset with:
  - Target variable (binary churn label)
  - Numerical features (age, tenure, charges, etc.)
  - Categorical features (service type, contract, etc.)

## 🛠️ Troubleshooting

### Import Error: anthropic
```bash
pip install anthropic
```

### ANTHROPIC_API_KEY Not Set
```bash
export ANTHROPIC_API_KEY="sk-..."
```

### PDF Generation Fails
```bash
# Install WeasyPrint dependencies
pip install weasyprint
```

### Memory Issues with SHAP
Reduce `shap_sample_size` in config.yaml

## 📚 References

- [SHAP Documentation](https://shap.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [APScheduler Documentation](https://apscheduler.readthedocs.io/)
- [Claude API Documentation](https://docs.anthropic.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

Created as a portfolio project demonstrating:
- Full ML pipeline engineering
- Advanced prompting techniques (Harness Engineering)
- Production-ready automation
- LLM integration for insights
- Professional data science practices

## 🎯 Future Enhancements

- [ ] Real-time prediction API (FastAPI)
- [ ] Interactive Streamlit dashboard
- [ ] Batch prediction with queue system
- [ ] Model drift detection
- [ ] A/B testing framework
- [ ] Customer retention strategy simulator
- [ ] Multi-model ensemble with voting
- [ ] Explainable AI dashboard (SHAP visualizations)

---

**Last Updated**: 2024
**Status**: Production Ready ✅
