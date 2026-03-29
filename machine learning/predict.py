import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

MODEL_PATH = os.path.join('ml_outputs', 'best_model.pkl')
FEATURES_PATH = os.path.join('ml_outputs', 'features.pkl')
TRAIN_CSV = 'cleaned_supply_chain.csv'

RAW_FEATURES = [
    'Type', 'Order Status', 'order_date', 'shipping_date',
    'Days for shipment (scheduled)', 'Shipping Mode', 'Sales',
    'Order Item Quantity', 'Order Item Discount', 'Order Item Discount Rate',
    'Order Item Product Price', 'Order Item Profit Ratio',
    'Order Profit Per Order', 'Benefit per order', 'profit_margin',
    'Product Price', 'Product Status', 'Category Name', 'Department Name',
    'Customer Id', 'Customer Segment', 'Customer City', 'Customer State',
    'Customer Country', 'Market', 'Order Region', 'Order Country',
    'Order City', 'Order State'
]

DROP_COLUMNS = [
    'Order Id', 'Product Name', 'order_date', 'shipping_date',
    'delivery_delay', 'Days for shipping (real)', 'Delivery Status',
    'Late_delivery_risk'
]

PRIOR_GROUPS = [
    'Order Region', 'Shipping Mode', 'Customer Segment',
    'Category Name', 'Order Country', 'Market'
]
CUM_GROUPS = ['Customer Id', 'Customer Country', 'Order Country']


def load_training_stats():
    df = pd.read_csv(TRAIN_CSV)
    overall_mean = df['is_late'].mean()
    high_value_threshold = df['Sales'].quantile(0.75)

    group_maps = {}
    for col in PRIOR_GROUPS + CUM_GROUPS:
        group_maps[col] = df.groupby(col)['is_late'].mean().to_dict()

    df_feature = df.copy()
    df_feature['order_date'] = pd.to_datetime(df_feature['order_date'])
    df_feature['shipping_date'] = pd.to_datetime(df_feature['shipping_date'])
    df_feature['order_weekday'] = df_feature['order_date'].dt.weekday
    df_feature['order_month'] = df_feature['order_date'].dt.month
    df_feature['order_day'] = df_feature['order_date'].dt.day
    df_feature['order_is_weekend'] = df_feature['order_weekday'].isin([5, 6]).astype(int)
    df_feature['scheduled_days'] = df_feature['Days for shipment (scheduled)']
    df_feature['actual_workdays'] = (df_feature['shipping_date'] - df_feature['order_date']).dt.days
    df_feature['delta_days'] = df_feature['actual_workdays'] - df_feature['scheduled_days']
    df_feature['delta_ratio'] = df_feature['delta_days'] / df_feature['scheduled_days'].replace(0, 1)

    for col in PRIOR_GROUPS:
        df_feature[f'{col}_prior_late_rate'] = df_feature.groupby(col)['is_late'].transform(lambda x: x.expanding().mean().shift(1))
        df_feature[f'{col}_prior_late_rate'] = df_feature[f'{col}_prior_late_rate'].fillna(overall_mean)

    for col in CUM_GROUPS:
        df_feature[f'{col}_cum_late_rate'] = df_feature.groupby(col)['is_late'].transform(lambda x: x.expanding().mean().shift(1))
        df_feature[f'{col}_cum_late_rate'] = df_feature[f'{col}_cum_late_rate'].fillna(overall_mean)

    df_feature['is_high_value'] = (df_feature['Sales'] > high_value_threshold).astype(int)
    df_feature['has_discount'] = (df_feature['Order Item Discount'] > 0).astype(int)

    df_feature = df_feature.drop(columns=[c for c in DROP_COLUMNS if c in df_feature.columns], errors='ignore')
    df_feature = df_feature.drop(columns=['Customer City', 'Customer State'], errors='ignore')
    X_train = df_feature.drop(columns=['is_late'])

    if os.path.exists(FEATURES_PATH):
        feature_order = pickle.load(open(FEATURES_PATH, 'rb'))
    else:
        feature_order = list(X_train.columns)

    label_maps = {}
    for col in X_train.select_dtypes(include=['object', 'category']).columns.tolist():
        values = X_train[col].fillna('Unknown').astype(str).unique().tolist()
        values = ['Unknown'] + [v for v in values if v != 'Unknown']
        label_maps[col] = {val: idx for idx, val in enumerate(values)}

    X_encoded = X_train.copy()
    for col, mapping in label_maps.items():
        X_encoded[col] = X_encoded[col].fillna('Unknown').astype(str).map(mapping).fillna(-1).astype(int)

    medians = X_encoded.median().to_dict()

    return {
        'group_maps': group_maps,
        'overall_mean': overall_mean,
        'high_value_threshold': high_value_threshold,
        'label_maps': label_maps,
        'medians': medians,
        'feature_order': feature_order,
    }


def parse_date(value, field_name):
    try:
        return pd.to_datetime(value)
    except Exception:
        raise ValueError(f'Invalid date for {field_name}: {value}')


def feature_engineering_inference(df, stats):
    df = df.copy()
    df['order_date'] = df['order_date'].apply(lambda v: parse_date(v, 'order_date'))
    df['shipping_date'] = df['shipping_date'].apply(lambda v: parse_date(v, 'shipping_date'))

    df['order_weekday'] = df['order_date'].dt.weekday
    df['order_month'] = df['order_date'].dt.month
    df['order_day'] = df['order_date'].dt.day
    df['order_is_weekend'] = df['order_weekday'].isin([5, 6]).astype(int)
    df['scheduled_days'] = df['Days for shipment (scheduled)']
    df['actual_workdays'] = (df['shipping_date'] - df['order_date']).dt.days
    df['delta_days'] = df['actual_workdays'] - df['scheduled_days']
    df['delta_ratio'] = df['delta_days'] / df['scheduled_days'].replace(0, 1)

    for col in PRIOR_GROUPS:
        df[f'{col}_prior_late_rate'] = df[col].map(stats['group_maps'][col]).fillna(stats['overall_mean'])

    for col in CUM_GROUPS:
        df[f'{col}_cum_late_rate'] = df[col].map(stats['group_maps'][col]).fillna(stats['overall_mean'])

    df['is_high_value'] = (df['Sales'] > stats['high_value_threshold']).astype(int)
    df['has_discount'] = (df['Order Item Discount'] > 0).astype(int)

    return df


def encode_and_impute(df, stats):
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors='ignore')
    df = df.drop(columns=['Customer City', 'Customer State'], errors='ignore')

    for col, mapping in stats['label_maps'].items():
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).map(mapping).fillna(-1).astype(int)

    for col, median in stats['medians'].items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(median)

    for col in stats['feature_order']:
        if col not in df.columns:
            df[col] = stats['medians'].get(col, 0)

    df = df[stats['feature_order']]
    return df


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_batch(input_path, output_path):
    df = pd.read_csv(input_path)
    stats = load_training_stats()
    df_fe = feature_engineering_inference(df, stats)
    X = encode_and_impute(df_fe, stats)

    model = load_model()
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    df_out = df.copy()
    df_out['predicted_late'] = y_pred
    df_out['late_probability'] = np.round(y_prob, 4)
    df_out.to_csv(output_path, index=False)
    print(f'Batch predictions saved to {output_path}')


def get_input_value(prompt, cast_type):
    while True:
        raw = input(prompt).strip()
        if raw == '':
            print('Input cannot be empty.')
            continue
        try:
            return cast_type(raw)
        except Exception:
            print(f'Invalid value: {raw}. Expected {cast_type.__name__}.')


def interactive_predict():
    print('Interactive single-order prediction')
    values = {}
    for feature in RAW_FEATURES:
        if feature in ['order_date', 'shipping_date']:
            values[feature] = input(f'{feature} (YYYY-MM-DD HH:MM:SS): ').strip()
            continue
        if feature in ['Sales', 'Order Item Discount', 'Order Item Discount Rate', 'Order Item Product Price',
                       'Order Item Profit Ratio', 'Order Profit Per Order', 'Benefit per order', 'profit_margin',
                       'Product Price']:
            values[feature] = float(input(f'{feature}: ').strip())
            continue
        if feature in ['Order Item Quantity', 'Product Status', 'Customer Id']:
            values[feature] = int(input(f'{feature}: ').strip())
            continue
        values[feature] = input(f'{feature}: ').strip()

    df = pd.DataFrame([values])
    stats = load_training_stats()
    df_fe = feature_engineering_inference(df, stats)
    X = encode_and_impute(df_fe, stats)

    model = load_model()
    y_pred = model.predict(X)[0]
    y_prob = model.predict_proba(X)[0, 1]

    print('\nPredicted late delivery:', int(y_pred))
    print('Late delivery probability:', round(float(y_prob), 4))


def main():
    parser = argparse.ArgumentParser(description='Predict late delivery risk')
    parser.add_argument('--input', type=str, help='CSV file with new orders')
    parser.add_argument('--output', type=str, default=os.path.join('ml_outputs', 'predictions_output.csv'), help='Output CSV file')
    args = parser.parse_args()

    if args.input:
        predict_batch(args.input, args.output)
    else:
        interactive_predict()


if __name__ == '__main__':
    main()
