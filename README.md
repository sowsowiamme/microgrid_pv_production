
# Microgrid PV Production Prediction ⚡🌞

A production-ready machine learning system for accurate photovoltaic power prediction and microgrid optimization.

## 🚀 Features

- **Multi-model PV Power Prediction** (Random Forest, LightGBM, XGBoost)
- **Advanced Feature Engineering** with weather and temporal features  
- **Time Series Validation** with proper train-test splits (specifically, try to avoid time leakage)/ try prediction with online learning 
- **Model Performance Tracking** with MLflow
- **REST API** for real-time predictions

## 📊 Project Status

- [x] Project structure established
- [x] Configuration system designed
- [ ] PV prediction models implementation
- [ ] Feature engineering pipeline
- [ ] Model deployment API

## 🛠️ Quick Start

```bash
git clone https://github.com/yourusername/microgrid_pv_production.git
cd microgrid_pv_production
pip install -r requirements.txt
python scripts/train_pv_model.py
