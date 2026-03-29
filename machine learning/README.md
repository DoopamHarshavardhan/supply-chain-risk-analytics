# Supply Chain Late Delivery Prediction

## Project overview
This project predicts late delivery risk for supply chain orders using machine learning. It trains a model on cleaned order data and exports predictions and model artifacts for reporting or deployment.

## File structure
- `supply_chain_ml.py` — improved training pipeline with feature engineering and model selection
- `predict.py` — inference script for batch and interactive single-order prediction
- `ml_outputs/` — saved model files, prediction exports, and summaries
- `cleaned_supply_chain.csv` — cleaned training data used to build the model

## How to run the training pipeline
```bash
python supply_chain_ml.py
```

## How to run batch inference
```bash
python predict.py --input new_orders.csv --output ml_outputs/predictions_output.csv
```

## How to run interactive single-order prediction
```bash
python predict.py
```

## Model details
- Models trained: CatBoost, LightGBM, HistGradientBoosting, RandomForest
- Best model selected by F1 score
- Reported accuracy: 99.45%

## Output files explanation
- `ml_outputs/best_model.pkl` — serialized best trained model for inference
- `ml_outputs/features.pkl` — ordered feature list used by the model
- `ml_outputs/predictions_v2.csv` — predictions for the training dataset from the final model
- `ml_outputs/model_summary_v2.csv` — final model metrics and selected best model details
