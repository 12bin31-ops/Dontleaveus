# ⚡ Quick Start Guide

Get the Customer Churn Prediction system running in 5 minutes.

## 1️⃣ Setup (2 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key for LLM insights
export ANTHROPIC_API_KEY="your-api-key-here"
```

## 2️⃣ Download Data (1 minute)

```bash
# Download Kaggle dataset automatically
python scripts/download_data.py

# Or create dummy data for testing
python scripts/download_data.py  # Choose option 3
```

## 3️⃣ Train Model (1 minute)

```bash
# Train all models and generate report
python scripts/train_model.py

# Or with custom data path
python scripts/train_model.py --data data/raw/your_data.csv
```

✅ **Output:**
- Trained model saved to `models/best_model.pkl`
- Report generated at `reports/churn_report_*.html`

## 4️⃣ Make Predictions (1 minute)

```bash
# Run inference on new customers
python scripts/run_inference.py --data data/raw/new_customers.csv

# Predictions saved to reports/predictions.csv
```

## 5️⃣ (Optional) Setup Automated Scheduling

```bash
# Run with scheduler
python scripts/train_model.py --setup-scheduler

# This will:
# - Retrain model every Monday 9 AM
# - Run inference daily at 8 AM  
# - Generate reports every Tuesday 10 AM
```

---

## 📊 What Happens

### Training Pipeline
1. **Loads** customer data
2. **Preprocesses** (scaling, encoding, SMOTE for imbalance)
3. **Trains** 4 ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)
4. **Optimizes** hyperparameters with Optuna
5. **Evaluates** with Precision, Recall, F1, ROC-AUC
6. **Explains** with SHAP for interpretability
7. **Generates insights** using Claude with Harness Engineering:
   - Chain of Thought (CoT)
   - Self-Consistency
   - ReAct (Reason-Act-Observe)
   - Memory Compaction
8. **Creates** HTML/PDF report with visualizations
9. **Identifies** high-risk customers
10. **Saves** best model for production

### Inference Pipeline
1. **Loads** new customer data
2. **Preprocesses** same as training
3. **Predicts** churn probability for each customer
4. **Ranks** customers by risk level
5. **Saves** predictions to CSV

---

## 📁 Generated Files

```
After training:
├── models/
│   ├── best_model.pkl           # Trained model
│   └── best_model_metadata.json  # Model info
├── reports/
│   ├── churn_report_*.html       # Interactive report
│   ├── churn_report_*.pdf        # PDF version
│   └── predictions.csv           # Inference results
└── logs/
    └── pipeline.log              # Execution logs
```

---

## 🎯 Example: Full Workflow

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-..."

# 2. Get data
python scripts/download_data.py

# 3. Train
python scripts/train_model.py

# 4. Make predictions
python scripts/run_inference.py --data data/raw/churn_data.csv

# 5. Check results
open reports/churn_report_*.html  # macOS
start reports/churn_report_*.html # Windows
```

---

## ⚙️ Configuration

Edit `config/config.yaml` for:

```yaml
# Model selection
models:
  classifiers:
    - name: "XGBoost"
      params:
        n_estimators: 100
        max_depth: 6

# Hyperparameter tuning
hyperparameter_tuning:
  enabled: true
  n_trials: 50

# Scheduling
scheduling:
  retrain:
    day_of_week: "mon"  # Monday
    hour: 9             # 9 AM
```

---

## 🐛 Troubleshooting

### `ImportError: No module named anthropic`
```bash
pip install anthropic
```

### `ANTHROPIC_API_KEY not found`
```bash
export ANTHROPIC_API_KEY="sk-..."
```

### Out of memory
```yaml
# In config.yaml
evaluation:
  shap_sample_size: 50  # Reduce from 100
```

### Data not found
```bash
python scripts/download_data.py
```

---

## 🚀 Docker Alternative

```bash
# Build image
docker build -t churn-prediction .

# Run
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/reports:/app/reports \
           churn-prediction

# Or with Docker Compose
docker-compose up
```

---

## 📈 Expected Results

**Model Performance (typical):**
- Accuracy: 80-85%
- Precision: 75-80%
- Recall: 60-70%
- F1 Score: 0.67-0.75
- ROC-AUC: 0.85-0.90

**Report Contents:**
- Executive summary with AI insights
- 5 model performance metrics cards
- Top 10 features driving churn
- Top 10 high-risk customers
- LLM-generated business insights
- Recommended interventions
- Visualizations (charts, confusion matrix, ROC curve)

---

## 💡 Next Steps

1. **Integrate with your data** - Replace `data/raw/churn_data.csv`
2. **Customize models** - Add/modify in `config.yaml`
3. **Deploy to production** - Use Docker + scheduling
4. **Monitor performance** - Check logs and reports
5. **Fine-tune LLM insights** - Modify prompts in `config.yaml`

---

## 📚 Learn More

- Full documentation: [README.md](README.md)
- Configuration guide: [config/config.yaml](config/config.yaml)
- API reference: Check docstrings in `src/`
- Harness Engineering techniques: See `src/insight_generator.py`

---

## ✅ Success Checklist

- [x] Python 3.8+ installed
- [x] Dependencies installed (`pip install -r requirements.txt`)
- [x] ANTHROPIC_API_KEY set
- [x] Data downloaded (`python scripts/download_data.py`)
- [x] Model trained (`python scripts/train_model.py`)
- [x] Report generated (check `reports/`)
- [x] Ready for production! 🚀

---

**Total time: ~5-10 minutes** ⏱️

Questions? Check README.md or open an issue on GitHub!
