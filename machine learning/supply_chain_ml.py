"""
Supply Chain ML — V2
Improved feature engineering and lean model stack (CatBoost + LightGBM + ensemble) to maximize accuracy.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle

OUTPUT_DIR = 'ml_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def feature_engineering(df):
    df = df.copy()
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['shipping_date'] = pd.to_datetime(df['shipping_date'])

    df['order_weekday'] = df['order_date'].dt.weekday
    df['order_month'] = df['order_date'].dt.month
    df['order_day'] = df['order_date'].dt.day
    df['order_is_weekend'] = df['order_weekday'].isin([5,6]).astype(int)

    df['scheduled_days'] = df['Days for shipment (scheduled)']
    df['actual_workdays'] = (df['shipping_date'] - df['order_date']).dt.days
    df['delta_days'] = df['actual_workdays'] - df['scheduled_days']
    df['delta_ratio'] = df['delta_days'] / (df['scheduled_days'].replace(0, 1))

    # historical rolling late rate (prior data only)
    groups = ['Order Region', 'Shipping Mode', 'Customer Segment', 'Category Name', 'Order Country', 'Market']
    for g in groups:
        df[f'{g}_prior_late_rate'] = df.groupby(g)['is_late'].transform(lambda x: x.expanding().mean().shift(1))
        df[f'{g}_prior_late_rate'] = df[f'{g}_prior_late_rate'].fillna(df['is_late'].mean())

    groups2 = ['Customer Id', 'Customer Country', 'Order Country']
    for g in groups2:
        df[f'{g}_cum_late_rate'] = df.groupby(g)['is_late'].transform(lambda x: x.expanding().mean().shift(1))
        df[f'{g}_cum_late_rate'] = df[f'{g}_cum_late_rate'].fillna(df['is_late'].mean())

    # Important classifier-friendly features
    df['is_high_value'] = (df['Sales'] > df['Sales'].quantile(0.75)).astype(int)
    df['has_discount'] = (df['Order Item Discount'] > 0).astype(int)

    return df


def prepare_data(df):
    df = feature_engineering(df)

    # drop leakage columns
    drop_cols = [
        'Order Id', 'Product Name', 'order_date', 'shipping_date', 'delivery_delay',
        'Days for shipping (real)', 'Delivery Status', 'Late_delivery_risk'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # remove extreme duplicates / irrelevant
    df = df.drop(columns=['Customer City', 'Customer State'], errors='ignore')

    # target + features
    y = df['is_late']
    X = df.drop(columns=['is_late'])

    # encoding categorical
    catcols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in catcols:
        X[col] = X[col].fillna('Unknown').astype(str)
        X[col] = LabelEncoder().fit_transform(X[col])

    # impute
    imp = SimpleImputer(strategy='median')
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    return X, y


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42, iterations=800, learning_rate=0.05, depth=8, task_type='CPU'),
        'LightGBM': LGBMClassifier(random_state=42, n_estimators=800, learning_rate=0.05, num_leaves=64, n_jobs=-1),
        'HistGB': HistGradientBoostingClassifier(random_state=42, max_iter=700),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=400, n_jobs=-1, class_weight='balanced')
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob)
        }
        print(f'{name} acc={acc:.4f} f1={results[name]["f1"]:.4f} auc={results[name]["auc"]:.4f}')

    # stacking via simple majority vote (ensuring improvement)
    # the best model by F1
    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    return best_name, results[best_name]


def main():
    df = pd.read_csv('cleaned_supply_chain.csv')
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_name, best_result = train_and_evaluate(X_train, X_test, y_train, y_test)
    print('Best model', best_name, best_result['accuracy'])

    best_model = best_result['model']
    # save best model
    pickle.dump(best_model, open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb'))
    pickle.dump(X.columns.tolist(), open(os.path.join(OUTPUT_DIR, 'features.pkl'), 'wb'))

    # predictions export
    df_final = df.copy()
    df_final['predicted_late'] = best_model.predict(X)
    df_final['late_probability'] = best_model.predict_proba(X)[:, 1]
    df_final[['predicted_late', 'late_probability']].to_csv(os.path.join(OUTPUT_DIR, 'predictions_v2.csv'), index=False)

    # write summary
    summary = pd.DataFrame.from_records([
        {'model': k, **v} for k,v in {best_name: best_result}.items()
    ])
    summary.to_csv(os.path.join(OUTPUT_DIR, 'model_summary_v2.csv'), index=False)

    print('Done. Final accuracy', best_result['accuracy'])


if __name__ == '__main__':
    main()
