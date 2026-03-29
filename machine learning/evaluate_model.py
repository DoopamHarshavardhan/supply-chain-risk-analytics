import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

OUTPUT_DIR = 'ml_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['shipping_date'] = pd.to_datetime(df['shipping_date'])

    df['order_weekday'] = df['order_date'].dt.weekday
    df['order_month'] = df['order_date'].dt.month
    df['order_day'] = df['order_date'].dt.day
    df['order_is_weekend'] = df['order_weekday'].isin([5, 6]).astype(int)

    df['scheduled_days'] = df['Days for shipment (scheduled)']
    df['actual_workdays'] = (df['shipping_date'] - df['order_date']).dt.days
    df['delta_days'] = df['actual_workdays'] - df['scheduled_days']
    df['delta_ratio'] = df['delta_days'] / df['scheduled_days'].replace(0, 1)

    groups = [
        'Order Region', 'Shipping Mode', 'Customer Segment',
        'Category Name', 'Order Country', 'Market'
    ]
    for g in groups:
        df[f'{g}_prior_late_rate'] = df.groupby(g)['is_late'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df[f'{g}_prior_late_rate'] = df[f'{g}_prior_late_rate'].fillna(df['is_late'].mean())

    groups2 = ['Customer Id', 'Customer Country', 'Order Country']
    for g in groups2:
        df[f'{g}_cum_late_rate'] = df.groupby(g)['is_late'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df[f'{g}_cum_late_rate'] = df[f'{g}_cum_late_rate'].fillna(df['is_late'].mean())

    df['is_high_value'] = (df['Sales'] > df['Sales'].quantile(0.75)).astype(int)
    df['has_discount'] = (df['Order Item Discount'] > 0).astype(int)

    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = feature_engineering(df)

    drop_cols = [
        'Order Id', 'Product Name', 'order_date', 'shipping_date', 'delivery_delay',
        'Days for shipping (real)', 'Delivery Status', 'Late_delivery_risk'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    df = df.drop(columns=['Customer City', 'Customer State'], errors='ignore')

    y = df['is_late']
    X = df.drop(columns=['is_late'])

    catcols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in catcols:
        X[col] = X[col].fillna('Unknown').astype(str)
        X[col] = LabelEncoder().fit_transform(X[col])

    imp = SimpleImputer(strategy='median')
    X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    return X, y


def save_plot_precision_recall(y_true, y_scores, output_path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, where='post', color='b', alpha=0.8)
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (AP = {ap:.4f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix_heatmap(cm, labels, output_path):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    df = pd.read_csv('cleaned_supply_chain.csv')
    X, y = prepare_data(df)

    features_path = os.path.join(OUTPUT_DIR, 'features.pkl')
    model_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')

    with open(features_path, 'rb') as f:
        feature_order = pickle.load(f)

    if set(feature_order) != set(X.columns):
        missing = [c for c in feature_order if c not in X.columns]
        extra = [c for c in X.columns if c not in feature_order]
        raise ValueError(
            f'Feature set mismatch: missing={missing}, extra={extra}.\n'
            f'Loaded features.pkl expects {len(feature_order)} columns, prepared data has {len(X.columns)}.'
        )

    X = X[feature_order]

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    class_counts = y.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, target_names=['On Time', 'Late'], zero_division=0)
    roc_auc = roc_auc_score(y_test, y_scores)

    pr_curve_path = os.path.join(OUTPUT_DIR, 'precision_recall_curve.png')
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    save_plot_precision_recall(y_test, y_scores, pr_curve_path)
    save_confusion_matrix_heatmap(cm, ['On Time', 'Late'], cm_path)

    summary_lines = [
        '=== Model Evaluation Summary ===',
        '',
        'Dataset class distribution:',
        f'  On Time (0): {int(class_counts.get(0, 0))}',
        f'  Late (1): {int(class_counts.get(1, 0))}',
        '',
        'Test set class distribution:',
        f'  On Time (0): {int(test_counts.get(0, 0))}',
        f'  Late (1): {int(test_counts.get(1, 0))}',
        '',
        'Confusion matrix (rows=true, cols=predicted):',
        str(cm),
        '',
        'Classification report:',
        report,
        f'ROC-AUC score: {roc_auc:.4f}',
        '',
        f'Precision-Recall curve saved to: {pr_curve_path}',
        f'Confusion matrix image saved to: {cm_path}',
        '',
        f'Feature count: {len(feature_order)}',
        f'Test set size: {len(X_test)}',
    ]

    report_file = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print('\n'.join(summary_lines))
    print(f'\nEvaluation report saved to: {report_file}')


if __name__ == '__main__':
    main()
