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

# 🔮 고객 이탈 예측 시스템 (Harness Engineering 적용)

머신러닝과 LLM을 결합하여 고객 이탈을 예측하고, 자동화된 비즈니스 인사이트를 생성하는 완전 자동화 시스템입니다.

## 🌟 핵심 기능

### 1. **머신러닝 기반 이탈 예측**
- 📊 **데이터 처리**
  - CSV 데이터 자동 로드 및 탐색
  - 결측치 처리
  - 범주형 변수 인코딩
  - 수치형 변수 정규화 (스케일링)
  - SMOTE를 이용한 불균형 데이터 처리

- 🤖 **모델 훈련**
  - 4가지 알고리즘 자동 비교
    - 로지스틱 회귀 (Logistic Regression)
    - 랜덤 포레스트 (Random Forest)
    - XGBoost
    - LightGBM
  - Optuna를 이용한 자동 하이퍼파라미터 튜닝
  - 5-fold 교차검증

- 📈 **모델 평가**
  - 정확도 (Accuracy)
  - 정밀도 (Precision)
  - 재현율 (Recall)
  - F1 점수
  - ROC-AUC
  - 혼동행렬 및 ROC 곡선
  - SHAP을 이용한 모델 해석

### 2. **🤖 Harness Engineering을 이용한 LLM 인사이트**

Claude API를 활용해 고급 프롬프팅 기법으로 비즈니스 인사이트를 자동 생성합니다.

**적용된 기법:**

1. **Chain of Thought (CoT)**
   - 단계별 추론 과정 제시
   - 복잡한 분석을 논리적으로 풀어냄

2. **Self-Consistency (자체 일관성)**
   - 여러 관점에서 동일 문제 분석
   - 비즈니스 관점 + 기술 관점 + 고객 세그먼트 관점
   - 각 관점을 통합하여 최종 인사이트 생성

3. **ReAct (Reason-Act-Observe)**
   - 추론 → 행동 → 관찰 의 반복
   - 패턴 발견 후 더 깊은 분석 진행

4. **Memory Compaction (메모리 압축)**
   - 긴 분석 결과를 3-5개 핵심 사항으로 압축
   - 경영진 요약용 간결한 요점 제시

### 3. **📝 자동화된 HTML/PDF 리포트 생성**

**리포트 포함 내용:**
1. 경영진 요약 (Executive Summary)
2. 모델 성능 지표 (시각적 카드)
3. 상위 피처 분석 (상위 10개)
4. 고위험 고객 식별 (상위 10명)
5. **AI 생성 인사이트**
   - 요약 인사이트
   - 피처 분석
   - 위험 고객 세그먼트
   - 압축된 핵심 사항
6. 권장 조치사항 (우선순위 표시)
7. 시각화 (차트, 그래프, 혼동행렬)

**출력 형식:**
- HTML (대시보드 스타일, 인터랙티브)
- PDF (정식 리포트)
- JSON (프로그래밍 접근)

### 4. **⏰ 완전 자동화 파이프라인**

APScheduler를 이용한 자동 스케줄링:
```
매주 월요일 09:00 → 모델 재훈련
매일 08:00 → 새 고객 데이터 추론
매주 화요일 10:00 → 리포트 자동 생성
```

### 5. **📦 프로덕션 배포 준비**

- Docker 컨테이너화
- Docker Compose로 다중 서비스 관리
- PostgreSQL 데이터베이스 (선택사항)
- Prometheus 모니터링 (선택사항)
- 완전한 에러 처리 및 로깅

---

## 📂 프로젝트 구조

```
customer-churn-prediction/
├── config/
│   └── config.yaml              # 전체 프로젝트 설정
├── data/
│   ├── raw/                     # 원본 데이터
│   └── processed/               # 전처리된 데이터
├── src/
│   ├── __init__.py
│   ├── utils.py                 # 유틸리티 함수
│   ├── data_loader.py           # 데이터 로드 & 전처리
│   ├── model_trainer.py         # 모델 훈련 & 하이퍼파라미터 튜닝
│   ├── model_evaluator.py       # 모델 평가 & SHAP 설명
│   ├── insight_generator.py     # LLM 인사이트 생성 (Harness Engineering)
│   └── report_generator.py      # 자동 리포트 생성
├── pipeline/
│   ├── __init__.py
│   ├── main.py                  # 메인 파이프라인
│   └── scheduler.py             # APScheduler 설정
├── models/                       # 학습된 모델 저장
├── reports/                      # 생성된 리포트
├── logs/
│   └── pipeline.log
├── scripts/
│   ├── train_model.py           # 훈련 스크립트
│   ├── run_inference.py         # 추론 스크립트
│   └── download_data.py         # 데이터 다운로드
├── templates/
│   └── report_template.html     # HTML 리포트 템플릿
├── tests/                        # 단위 테스트
├── Dockerfile                   # Docker 이미지 설정
├── docker-compose.yml           # Docker Compose 설정
├── requirements.txt             # Python 의존성
├── README.md                    # 영문 설명서
├── README_KO.md                 # 한글 설명서 (이 파일)
├── QUICKSTART.md                # 빠른 시작 가이드
└── PROJECT_SUMMARY.md           # 프로젝트 요약
```

---

## 🚀 빠른 시작 (5분)

### 필수 요구사항
- Python 3.8 이상
- pip 또는 conda
- Docker (선택사항)

### 1. 설치 및 준비 (2분)

```bash
# 저장소 클론
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# API 키 설정 (LLM 인사이트용)
export ANTHROPIC_API_KEY="sk-..."  # Windows: set ANTHROPIC_API_KEY=sk-...
```

### 2. 데이터 준비 (1분)

```bash
# Kaggle 데이터 다운로드 또는 더미 데이터 생성
python scripts/download_data.py

# 선택지:
# 1) Kaggle API 사용
# 2) 공개 URL에서 다운로드
# 3) 테스트용 더미 데이터 생성
```

### 3. 모델 훈련 (1분)

```bash
# 전체 훈련 파이프라인 실행
python scripts/train_model.py
```

**자동으로 수행되는 작업:**
- ✅ 데이터 탐색 및 전처리
- ✅ 4개 모델 훈련 및 비교
- ✅ 최고 성능 모델 자동 선택
- ✅ SHAP으로 모델 해석
- ✅ **Claude API를 이용한 LLM 인사이트 생성**
- ✅ HTML/PDF 리포트 생성
- ✅ 최적 모델 저장

**출력 결과:**
```
✅ TRAINING COMPLETE
================================================
Best Model: XGBoost
Model Metrics:
  accuracy: 0.8234
  precision: 0.7856
  recall: 0.6543
  f1: 0.7143
  roc_auc: 0.8754

Model saved to: models/best_model.pkl
Report saved to: reports/churn_report_20240805_120000.html
```

### 4. 새 고객 예측 (1분)

```bash
# 새로운 고객 데이터에 대한 이탈 예측
python scripts/run_inference.py --data data/raw/new_customers.csv
```

**출력:**
```
INFERENCE COMPLETE
================================================
Predictions Summary:
Total customers: 150

Risk Distribution:
High   : 23 (15.3%)
Medium : 45 (30.0%)
Low    : 82 (54.7%)

Churn Probability Statistics:
  Mean: 0.3245
  Min: 0.0123
  Max: 0.9876
  Median: 0.2654

Predictions saved to: reports/predictions.csv

🚨 Top 10 High-Risk Customers:
    churn_probability  risk_level
0   0.9856            HIGH
1   0.9234            HIGH
2   0.8765            HIGH
...
```

---

## ⚙️ 설정 (config.yaml)

### 모델 선택 및 파라미터

```yaml
models:
  classifiers:
    - name: "Logistic Regression"
      type: "LogisticRegression"
      params:
        max_iter: 1000
        random_state: 42
        
    - name: "Random Forest"
      type: "RandomForestClassifier"
      params:
        n_estimators: 100
        max_depth: 10
        
    - name: "XGBoost"
      type: "XGBClassifier"
      params:
        n_estimators: 100
        max_depth: 6
        learning_rate: 0.1
```

### 하이퍼파라미터 튜닝

```yaml
hyperparameter_tuning:
  enabled: true          # 활성화/비활성화
  method: "optuna"       # optuna, grid, random
  n_trials: 50          # 시도 횟수
  timeout: 600          # 타임아웃 (초)
  cv_folds: 5           # 교차검증 폴드
```

### LLM 설정 (Harness Engineering)

```yaml
llm:
  provider: "anthropic"
  model: "claude-opus-4-6"
  max_tokens: 2000
  temperature: 0.7
  
  harness_engineering:
    enabled: true
    techniques:
      - chain_of_thought
      - self_consistency
      - react
      - memory_compaction
```

### 자동화 스케줄링

```yaml
scheduling:
  enabled: true
  
  retrain:              # 모델 재훈련
    trigger: "cron"
    day_of_week: "mon"  # 월요일
    hour: 9             # 09:00
    
  inference:            # 추론 실행
    trigger: "cron"
    day_of_week: "0-6"  # 매일
    hour: 8
    
  report:               # 리포트 생성
    trigger: "cron"
    day_of_week: "1"    # 화요일
    hour: 10
```

---

## 🎯 파이프라인 동작 흐름

### 훈련 파이프라인

```
1. 데이터 로드
   ↓
2. 탐색적 데이터 분석 (EDA)
   ↓
3. 데이터 전처리
   (결측치 처리, 인코딩, 스케일링, SMOTE)
   ↓
4. 훈련/테스트 분할
   ↓
5. 모델 훈련 (4개 알고리즘 병렬)
   ↓
6. 하이퍼파라미터 최적화 (Optuna)
   ↓
7. 모델 평가
   (정확도, 정밀도, 재현율, F1, ROC-AUC)
   ↓
8. SHAP으로 피처 중요도 분석
   ↓
9. LLM 인사이트 생성 (Harness Engineering)
   - Chain of Thought
   - Self-Consistency
   - ReAct
   - Memory Compaction
   ↓
10. 시각화 생성
    (혼동행렬, ROC 곡선, 피처 중요도)
   ↓
11. HTML/PDF 리포트 생성
   ↓
12. 최고 성능 모델 저장
```

### 추론 파이프라인

```
1. 새 고객 데이터 로드
   ↓
2. 훈련과 동일한 방식으로 전처리
   ↓
3. 저장된 모델 로드
   ↓
4. 각 고객의 이탈 확률 예측
   ↓
5. 고위험 고객 식별
   ↓
6. CSV로 결과 저장
   ↓
7. 알림 및 리포트 생성
```

---

## 📊 모델 비교

| 모델 | 장점 | 단점 | 추천 |
|------|------|------|------|
| 로지스틱 회귀 | 빠름, 해석 용이 | 정확도 낮음 | 기준선 |
| 랜덤 포레스트 | 좋은 균형 | 훈련 느림 | 일반용 |
| XGBoost | 높은 정확도 | 튜닝 복잡 | ⭐ 추천 |
| LightGBM | 빠름, 메모리 효율 | 과적합 경향 | 대용량 데이터 |

---

## 📈 평가 지표 설명

- **정확도 (Accuracy)**: 전체 예측의 정확성
- **정밀도 (Precision)**: 이탈 예측이 맞을 확률
- **재현율 (Recall)**: 실제 이탈 고객을 찾는 비율
- **F1 점수**: 정밀도와 재현율의 조화평균 (이 프로젝트의 주요 지표)
- **ROC-AUC**: 분류 모델의 전체 성능

---

## 💡 사용 예시

### Python 코드로 직접 사용

```python
from pipeline.main import ChurnPredictionPipeline

# 파이프라인 초기화
pipeline = ChurnPredictionPipeline(config_path='config/config.yaml')

# 훈련 실행
results = pipeline.run_training_pipeline(
    data_path='data/raw/churn_data.csv'
)

# 결과 확인
print(f"최고 성능 모델: {results['best_model_name']}")
print(f"F1 점수: {results['metrics']['f1']:.4f}")
print(f"생성된 리포트: {results['report_path']}")
```

### CLI 명령어

```bash
# 모델 훈련 (기본 설정)
python scripts/train_model.py

# 커스텀 데이터로 훈련
python scripts/train_model.py --data data/raw/my_data.csv

# 추론 (새 고객 예측)
python scripts/run_inference.py --data data/raw/new_customers.csv

# 리포트 생성만 수행
python scripts/train_model.py --generate-report

# 자동화 스케줄링 시작
python scripts/train_model.py --setup-scheduler
```

---

## 🐳 Docker로 실행

### 단일 컨테이너 실행

```bash
# 이미지 빌드
docker build -t customer-churn-prediction .

# 컨테이너 실행
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
           -v $(pwd)/data:/app/data \
           -v $(pwd)/reports:/app/reports \
           customer-churn-prediction
```

### Docker Compose 사용 (권장)

```bash
# 전체 서비스 시작 (앱 + 데이터베이스 + 모니터링)
docker-compose up

# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f app

# 중지
docker-compose down
```

---

## 📊 기대 결과

### 모델 성능 (일반적)
- **정확도**: 80-85%
- **정밀도**: 75-80%
- **재현율**: 60-70%
- **F1 점수**: 0.67-0.75
- **ROC-AUC**: 0.85-0.90

### 생성되는 리포트
- **파일**: `reports/churn_report_YYYYMMDD_HHMMSS.html`
- **크기**: 15-20페이지 (시각화 포함)
- **생성 시간**: 2-3분 (LLM 포함)
- **포함 내용**:
  - 경영진 요약
  - 5가지 성능 지표
  - 상위 10개 피처
  - 고위험 고객 10명
  - **AI 생성 비즈니스 인사이트**
  - 권장 조치사항
  - 차트 및 시각화

---

## 🧪 테스트

```bash
# 모든 테스트 실행
pytest tests/

# 커버리지와 함께 실행
pytest --cov=src tests/

# 특정 테스트만 실행
pytest tests/test_data_loader.py -v
```

---

## 🔧 트러블슈팅

### 문제 1: `ImportError: No module named 'anthropic'`
```bash
pip install anthropic
```

### 문제 2: `ANTHROPIC_API_KEY not found`
```bash
# macOS/Linux
export ANTHROPIC_API_KEY="sk-..."

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-..."
```

### 문제 3: 메모리 부족 오류
```yaml
# config.yaml 수정
evaluation:
  shap_sample_size: 50  # 100에서 50으로 감소

hyperparameter_tuning:
  n_trials: 20  # 50에서 20으로 감소
```

### 문제 4: 데이터 파일을 찾을 수 없음
```bash
# 데이터 다운로드 재실행
python scripts/download_data.py

# 또는 수동으로 CSV 파일을 data/raw/ 폴더에 배치
```

### 문제 5: PDF 생성 실패
```bash
# WeasyPrint 의존성 설치
pip install weasyprint

# Linux의 경우 추가 시스템 패키지 필요
sudo apt-get install libpango-1.0-0 libpango-gobject-0
```

---

## 📚 참고 자료

- [SHAP 공식 문서](https://shap.readthedocs.io/)
- [Optuna 문서](https://optuna.readthedocs.io/)
- [APScheduler 문서](https://apscheduler.readthedocs.io/)
- [Claude API 문서](https://docs.anthropic.com/)
- [XGBoost 문서](https://xgboost.readthedocs.io/)
- [Scikit-learn 문서](https://scikit-learn.org/)

---

## 🤝 기여 방법

개선 사항이 있으신가요? 환영합니다!

1. Fork 하기
2. Feature 브랜치 생성 (`git checkout -b feature/improvement`)
3. 변경사항 커밋 (`git commit -am 'Add improvement'`)
4. 브랜치 푸시 (`git push origin feature/improvement`)
5. Pull Request 생성

---

## 📄 라이선스

MIT 라이선스 - 자세한 내용은 LICENSE 파일 참조

---

## 👤 개발자

포트폴리오 프로젝트로 다음을 시연합니다:
- ✅ 완전한 ML 파이프라인 구축
- ✅ LLM 통합 및 고급 프롬프팅 기법
- ✅ 프로덕션급 자동화 시스템
- ✅ 전문적인 소프트웨어 엔지니어링

---

## 🎯 향후 계획

- [ ] FastAPI 기반 실시간 예측 API
- [ ] Streamlit 대시보드 구현
- [ ] 배치 예측 큐 시스템
- [ ] 모델 드리프트 감지
- [ ] A/B 테스트 프레임워크
- [ ] 고객 유지 전략 시뮬레이터
- [ ] 앙상블 모델 (다중 모델 투표)
- [ ] SHAP 대시보드 구현

---

