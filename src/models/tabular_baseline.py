import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, root_mean_squared_error, r2_score
)

def prepare_features(df: pd.DataFrame):
    df = df.copy()
    
    # 1. Cyclical temporal features from the prediction anchor
    df['order_hour'] = df['order_timestamp'].dt.hour
    df['order_dayofweek'] = df['order_timestamp'].dt.dayofweek
    df['order_month'] = df['order_timestamp'].dt.month
    
    # 2. Select valid point-in-time features
    feature_cols = [
        'Days for shipment (scheduled)',
        'Shipping Mode',
        'order_hour',
        'order_dayofweek',
        'order_month',
        'Order Item Quantity',
        'Order Item Product Price',
        'Order Item Discount Rate',
        'Category Name',
        'Department Name',
        'Customer Segment',
        'Customer State',
        'Customer Country',
        'Order Region',
        'Order Country',
        'Order State'
    ]
    
    # Intersect with existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy()
    
    # Convert categorical strings to category dtype for native LightGBM/XGBoost handling
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')
        
    return X

def run_benchmarks():
    print("[*] Loading chronological processed parquet files...")
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)
    X_test = prepare_features(test_df)
    
    # Classification targets
    y_train_clf = train_df['target_is_delayed']
    y_val_clf = val_df['target_is_delayed']
    y_test_clf = test_df['target_is_delayed']
    
    # Regression targets
    y_train_reg = train_df['target_delay_days']
    y_val_reg = val_df['target_delay_days']
    y_test_reg = test_df['target_delay_days']
    
    print(f"[+] Features prepared. Number of features: {X_train.shape[1]}")
    
    # ==========================================
    # 1. CLASSIFICATION TASK (Late vs On-time)
    # ==========================================
    print("\n" + "="*50)
    print("TASK 1: CLASSIFICATION (Point-in-Time Prediction)")
    print("="*50)
    
    # LightGBM Classifier
    print("[*] Training LightGBM Classifier...")
    clf_lgb = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
    clf_lgb.fit(
        X_train, y_train_clf,
        eval_set=[(X_val, y_val_clf)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    y_pred_clf = clf_lgb.predict(X_test)
    y_prob_clf = clf_lgb.predict_proba(X_test)[:, 1]
    
    print("\n--- LightGBM Classification Results (Held-Out Test Set) ---")
    print(f"Accuracy : {accuracy_score(y_test_clf, y_pred_clf):.4f}")
    print(f"Precision: {precision_score(y_test_clf, y_pred_clf):.4f}")
    print(f"Recall   : {recall_score(y_test_clf, y_pred_clf):.4f}")
    print(f"Macro F1 : {f1_score(y_test_clf, y_pred_clf, average='macro'):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test_clf, y_prob_clf):.4f}")
    
    # ==========================================
    # 2. REGRESSION TASK (Delay Duration Days)
    # ==========================================
    print("\n" + "="*50)
    print("TASK 2: REGRESSION (Delay Duration in Days)")
    print("="*50)
    
    # LightGBM Regressor
    print("[*] Training LightGBM Regressor...")
    reg_lgb = lgb.LGBMRegressor(n_estimators=300, random_state=42, verbose=-1)
    reg_lgb.fit(
        X_train, y_train_reg,
        eval_set=[(X_val, y_val_reg)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    y_pred_reg = reg_lgb.predict(X_test)
    
    print("\n--- LightGBM Regression Results (Held-Out Test Set) ---")
    print(f"MAE  : {mean_absolute_error(y_test_reg, y_pred_reg):.4f} days")
    print(f"RMSE : {root_mean_squared_error(y_test_reg, y_pred_reg):.4f} days")
    print(f"R²   : {r2_score(y_test_reg, y_pred_reg):.4f}")
    print("="*50)

if __name__ == "__main__":
    run_benchmarks()