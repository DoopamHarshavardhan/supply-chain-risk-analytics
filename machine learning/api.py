import os
import pickle
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer

MODEL_PATH = os.path.join('ml_outputs', 'best_model.pkl')
FEATURES_PATH = os.path.join('ml_outputs', 'features.pkl')
DATA_PATH = 'cleaned_supply_chain.csv'

app = FastAPI(title='Supply Chain Prediction API', version='1.0')

startup_error: Optional[str] = None
model = None
feature_order: List[str] = []
historical_df: Optional[pd.DataFrame] = None
label_maps: Dict[str, Dict[str, int]] = {}
imputer: Optional[SimpleImputer] = None
numeric_fields: List[str] = []


class OrderInput(BaseModel):
    order_id: Optional[str | int] = Field(default='Unknown', alias='Order Id')
    order_type: str = Field(default='Unknown', alias='Type')
    order_status: str = Field(default='Unknown', alias='Order Status')
    order_date: date = Field(default_factory=date.today, alias='order_date')
    shipping_date: date = Field(default_factory=date.today, alias='shipping_date')
    days_for_shipping_real: float = Field(default=0.0, alias='Days for shipping (real)')
    days_for_shipment_scheduled: float = Field(default=1.0, alias='Days for shipment (scheduled)')
    delivery_delay: float = Field(default=0.0, alias='delivery_delay')
    is_late: Optional[int] = Field(default=0, alias='is_late')
    delivery_status: str = Field(default='Unknown', alias='Delivery Status')
    late_delivery_risk: float = Field(default=0.0, alias='Late_delivery_risk')
    shipping_mode: str = Field(default='Unknown', alias='Shipping Mode')
    sales: float = Field(default=0.0, alias='Sales')
    order_item_quantity: float = Field(default=0.0, alias='Order Item Quantity')
    order_item_discount: float = Field(default=0.0, alias='Order Item Discount')
    order_item_discount_rate: float = Field(default=0.0, alias='Order Item Discount Rate')
    order_item_product_price: float = Field(default=0.0, alias='Order Item Product Price')
    order_item_profit_ratio: float = Field(default=0.0, alias='Order Item Profit Ratio')
    order_profit_per_order: float = Field(default=0.0, alias='Order Profit Per Order')
    benefit_per_order: float = Field(default=0.0, alias='Benefit per order')
    profit_margin: float = Field(default=0.0, alias='profit_margin')
    product_name: str = Field(default='Unknown', alias='Product Name')
    product_price: float = Field(default=0.0, alias='Product Price')
    product_status: str | int = Field(default='Unknown', alias='Product Status')
    category_name: str = Field(default='Unknown', alias='Category Name')
    department_name: str = Field(default='Unknown', alias='Department Name')
    customer_id: str | int = Field(default=0, alias='Customer Id')
    customer_segment: str = Field(default='Unknown', alias='Customer Segment')
    customer_city: str = Field(default='Unknown', alias='Customer City')
    customer_state: str = Field(default='Unknown', alias='Customer State')
    customer_country: str = Field(default='Unknown', alias='Customer Country')
    market: str = Field(default='Unknown', alias='Market')
    order_region: str = Field(default='Unknown', alias='Order Region')
    order_country: str = Field(default='Unknown', alias='Order Country')
    order_city: str = Field(default='Unknown', alias='Order City')
    order_state: str = Field(default='Unknown', alias='Order State')

    class Config:
        validate_by_name = True
        extra = 'ignore'


class PredictionResult(BaseModel):
    predicted_late: int
    late_probability: float
    on_time_probability: float


class BatchPredictionResponse(BaseModel):
    total: int
    late_count: int
    on_time_count: int
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_count: int
    training_rows: int


def _ensure_artifacts() -> None:
    global startup_error, model, feature_order, historical_df, label_maps, imputer

    if startup_error:
        raise RuntimeError(startup_error)

    if model is None or historical_df is None or not feature_order or imputer is None:
        raise RuntimeError('API startup has not completed successfully')


def _load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def _build_label_maps(X: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    mappings: Dict[str, Dict[str, int]] = {}
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        values = sorted(set(X[col].fillna('Unknown').astype(str).tolist()) | {'Unknown'})
        mappings[col] = {value: idx for idx, value in enumerate(values)}
    return mappings


def _label_encode_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col, mapping in label_maps.items():
        if col not in X.columns:
            continue
        X[col] = X[col].fillna('Unknown').astype(str).map(mapping)
        if 'Unknown' in mapping:
            X[col] = X[col].fillna(mapping['Unknown'])
        else:
            X[col] = X[col].fillna(0)
        X[col] = X[col].astype(int)
    return X


def _feature_engineering(df: pd.DataFrame, base_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.copy()
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])
    if 'shipping_date' in df.columns:
        df['shipping_date'] = pd.to_datetime(df['shipping_date'])

    if base_df is not None:
        base = base_df.copy()
        base['order_date'] = pd.to_datetime(base['order_date'])
        base['shipping_date'] = pd.to_datetime(base['shipping_date'])
        combined = pd.concat([base, df], ignore_index=True)
        sales_source = base['Sales']
    else:
        combined = df
        sales_source = combined['Sales']

    combined['order_weekday'] = combined['order_date'].dt.weekday
    combined['order_month'] = combined['order_date'].dt.month
    combined['order_day'] = combined['order_date'].dt.day
    combined['order_is_weekend'] = combined['order_weekday'].isin([5, 6]).astype(int)

    combined['scheduled_days'] = combined['Days for shipment (scheduled)']
    combined['actual_workdays'] = (combined['shipping_date'] - combined['order_date']).dt.days
    combined['delta_days'] = combined['actual_workdays'] - combined['scheduled_days']
    combined['delta_ratio'] = combined['delta_days'] / combined['scheduled_days'].replace(0, 1)

    sales_threshold = sales_source.quantile(0.75)
    combined['is_high_value'] = (combined['Sales'] > sales_threshold).astype(int)
    combined['has_discount'] = (combined['Order Item Discount'] > 0).astype(int)

    groups = [
        'Order Region',
        'Shipping Mode',
        'Customer Segment',
        'Category Name',
        'Order Country',
        'Market',
    ]
    for g in groups:
        combined[f'{g}_prior_late_rate'] = combined.groupby(g)['is_late'].transform(
            lambda values: values.expanding().mean().shift(1)
        )
        combined[f'{g}_prior_late_rate'] = combined[f'{g}_prior_late_rate'].fillna(combined['is_late'].mean())

    groups2 = ['Customer Id', 'Customer Country', 'Order Country']
    for g in groups2:
        combined[f'{g}_cum_late_rate'] = combined.groupby(g)['is_late'].transform(
            lambda values: values.expanding().mean().shift(1)
        )
        combined[f'{g}_cum_late_rate'] = combined[f'{g}_cum_late_rate'].fillna(combined['is_late'].mean())

    if base_df is not None:
        return combined.iloc[len(base_df):].reset_index(drop=True)

    return combined.reset_index(drop=True)


def _prepare_inputs(raw_rows: List[Dict]) -> pd.DataFrame:
    if not raw_rows:
        raise ValueError('No order data provided')

    data = pd.DataFrame(raw_rows)
    for column, default in [
        ('Order Id', 0),
        ('Type', 'Unknown'),
        ('Order Status', 'Unknown'),
        ('order_date', pd.Timestamp(date.today())),
        ('shipping_date', pd.Timestamp(date.today())),
        ('Days for shipping (real)', 0.0),
        ('Days for shipment (scheduled)', 1.0),
        ('delivery_delay', 0.0),
        ('is_late', 0.0),
        ('Delivery Status', 'Unknown'),
        ('Late_delivery_risk', 0.0),
        ('Shipping Mode', 'Unknown'),
        ('Sales', 0.0),
        ('Order Item Quantity', 0.0),
        ('Order Item Discount', 0.0),
        ('Order Item Discount Rate', 0.0),
        ('Order Item Product Price', 0.0),
        ('Order Item Profit Ratio', 0.0),
        ('Order Profit Per Order', 0.0),
        ('Benefit per order', 0.0),
        ('profit_margin', 0.0),
        ('Product Name', 'Unknown'),
        ('Product Price', 0.0),
        ('Product Status', 0),
        ('Category Name', 'Unknown'),
        ('Department Name', 'Unknown'),
        ('Customer Id', 0),
        ('Customer Segment', 'Unknown'),
        ('Customer City', 'Unknown'),
        ('Customer State', 'Unknown'),
        ('Customer Country', 'Unknown'),
        ('Market', 'Unknown'),
        ('Order Region', 'Unknown'),
        ('Order Country', 'Unknown'),
        ('Order City', 'Unknown'),
        ('Order State', 'Unknown'),
    ]:
        if column not in data.columns:
            data[column] = default

    numeric_defaults = {
        'Days for shipping (real)': 0.0,
        'Days for shipment (scheduled)': 1.0,
        'delivery_delay': 0.0,
        'is_late': np.nan,
        'Late_delivery_risk': 0.0,
        'Sales': 0.0,
        'Order Item Quantity': 0.0,
        'Order Item Discount': 0.0,
        'Order Item Discount Rate': 0.0,
        'Order Item Product Price': 0.0,
        'Order Item Profit Ratio': 0.0,
        'Order Profit Per Order': 0.0,
        'Benefit per order': 0.0,
        'profit_margin': 0.0,
        'Product Price': 0.0,
        'Product Status': 0,
        'Customer Id': 0,
    }
    for column, default in numeric_defaults.items():
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce').fillna(default)

    data['is_late'] = np.nan
    engineered = _feature_engineering(data, base_df=historical_df)
    drop_columns = [
        'Order Id',
        'Product Name',
        'order_date',
        'shipping_date',
        'delivery_delay',
        'Days for shipping (real)',
        'Delivery Status',
        'Late_delivery_risk',
        'Customer City',
        'Customer State',
    ]
    engineered = engineered.drop(columns=[c for c in drop_columns if c in engineered.columns], errors='ignore')
    engineered = _label_encode_frame(engineered)
    engineered = pd.DataFrame(imputer.transform(engineered), columns=engineered.columns)

    missing_cols = [c for c in feature_order if c not in engineered.columns]
    for col in missing_cols:
        engineered[col] = 0

    engineered = engineered[feature_order]
    return engineered


@app.on_event('startup')
def startup_event() -> None:
    global startup_error, model, feature_order, historical_df, label_maps, imputer

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model file not found at {MODEL_PATH}')
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f'Feature file not found at {FEATURES_PATH}')
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f'Historical data file not found at {DATA_PATH}')

        historical_df = pd.read_csv(DATA_PATH)
        feature_order = _load_pickle(FEATURES_PATH)
        model = _load_pickle(MODEL_PATH)

        if not isinstance(feature_order, list):
            feature_order = list(feature_order)

        training_frame = _feature_engineering(historical_df)
        numeric_fields = [
            c for c in training_frame.columns
            if pd.api.types.is_numeric_dtype(training_frame[c])
        ]
        drop_columns = [
            'Order Id',
            'Product Name',
            'order_date',
            'shipping_date',
            'delivery_delay',
            'Days for shipping (real)',
            'Delivery Status',
            'Late_delivery_risk',
            'Customer City',
            'Customer State',
        ]
        training_frame = training_frame.drop(columns=[c for c in drop_columns if c in training_frame.columns], errors='ignore')
        label_maps = _build_label_maps(training_frame)
        training_encoded = _label_encode_frame(training_frame)
        imputer = SimpleImputer(strategy='median')
        imputer.fit(training_encoded)

    except Exception as exc:
        startup_error = str(exc)


@app.get('/health', response_model=HealthResponse)
def health() -> Dict[str, object]:
    if startup_error:
        return {
            'status': 'error',
            'model_loaded': False,
            'feature_count': 0,
            'training_rows': 0,
        }
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'feature_count': len(feature_order),
        'training_rows': len(historical_df) if historical_df is not None else 0,
    }


@app.post('/predict', response_model=PredictionResult)
def predict(order: OrderInput) -> PredictionResult:
    if startup_error:
        raise HTTPException(status_code=503, detail=f'API startup failed: {startup_error}')

    raw = order.dict(by_alias=True)
    features = _prepare_inputs([raw])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return PredictionResult(
        predicted_late=prediction,
        late_probability=float(probabilities[1]),
        on_time_probability=float(probabilities[0]),
    )


@app.post('/predict/batch', response_model=BatchPredictionResponse)
def predict_batch(orders: List[OrderInput]) -> BatchPredictionResponse:
    if startup_error:
        raise HTTPException(status_code=503, detail=f'API startup failed: {startup_error}')

    raws = [order.dict(by_alias=True) for order in orders]
    features = _prepare_inputs(raws)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = [
        PredictionResult(
            predicted_late=int(prediction),
            late_probability=float(prob[1]),
            on_time_probability=float(prob[0]),
        )
        for prediction, prob in zip(predictions, probabilities)
    ]

    late_count = int((predictions == 1).sum())
    on_time_count = int((predictions == 0).sum())

    return BatchPredictionResponse(
        total=len(results),
        late_count=late_count,
        on_time_count=on_time_count,
        predictions=results,
    )
