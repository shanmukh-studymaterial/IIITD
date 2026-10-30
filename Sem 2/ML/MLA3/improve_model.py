import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import warnings

warnings.filterwarnings('ignore')

def load_data():
    train = pd.read_csv("train_assignment.csv")
    test = pd.read_csv("test_assignment.csv")
    return train, test

def engineer_features(df):
    # Success ratios
    df['cu1_success_rate'] = df['cu1_approved'] / df['cu1_enrolled']
    df['cu2_success_rate'] = df['cu2_approved'] / df['cu2_enrolled']
    
    # Handle division by zero
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    # Total performance
    df['total_approved'] = df['cu1_approved'] + df['cu2_approved']
    df['total_enrolled'] = df['cu1_enrolled'] + df['cu2_enrolled']
    df['total_success_rate'] = df['total_approved'] / df['total_enrolled']
    
    # Grade progress
    df['grade_diff'] = df['cu2_grade'] - df['cu1_grade']
    df['avg_grade'] = (df['cu1_grade'] + df['cu2_grade']) / 2
    
    # Effort/Engagement
    df['total_evaluations'] = df['cu1_evaluations'] + df['cu2_evaluations']
    
    # Dropped units
    df['cu1_dropped'] = df['cu1_enrolled'] - df['cu1_approved']
    df['cu2_dropped'] = df['cu2_enrolled'] - df['cu2_approved']
    df['total_dropped'] = df['cu1_dropped'] + df['cu2_dropped']
    
    # Financial indicators
    df['debt_and_tuition'] = df['debtor_flag'].astype(str) + "_" + df['tuition_up_to_date_flag'].astype(str)
    
    return df

def preprocess(train, test):
    train = engineer_features(train)
    test = engineer_features(test)
    
    X = train.drop(columns=['id', 'Target'])
    y = train['Target']
    test_X = test.drop(columns=['id'])
    test_ids = test['id']
    
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    
    # Identify categorical columns (mostly flags and codes)
    cat_cols = [col for col in X.columns if 'code' in col or 'flag' in col or col == 'debt_and_tuition']
    
    # Use OrdinalEncoder to handle unseen categories in test
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
        test_X[cat_cols] = encoder.transform(test_X[cat_cols].astype(str))
            
    return X, y, test_X, test_ids, target_encoder

def objective_lgb(trial, X, y):
    params = {
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'verbosity': -1
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
        
    return np.mean(scores)

def main():
    print("Loading data...")
    train, test = load_data()
    
    print("Preprocessing and Feature Engineering...")
    X, y, test_X, test_ids, target_encoder = preprocess(train, test)
    
    # Using pre-tuned high-performance params to save time, 
    # but I could run Optuna if I wanted.
    
    lgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.02,
        'num_leaves': 64,
        'max_depth': 10,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': -1
    }
    
    xgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.02,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    cat_params = {
        'n_estimators': 1000,
        'learning_rate': 0.02,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU'
    }
    
    models = [
        ('lgb', lgb.LGBMClassifier(**lgb_params)),
        ('xgb', xgb.XGBClassifier(**xgb_params)),
        ('cat', CatBoostClassifier(**cat_params))
    ]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(test_X), 3))
    
    print("Starting Cross-Validation and Ensembling (3 Folds)...")
    for name, model in models:
        print(f"Training {name}...")
        curr_oof = np.zeros((len(X), 3))
        curr_test = np.zeros((len(test_X), 3))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold+1}...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            curr_oof[val_idx] = model.predict_proba(X_val)
            curr_test += model.predict_proba(test_X) / 3
            
        acc = accuracy_score(y, np.argmax(curr_oof, axis=1))
        print(f"{name} CV Accuracy: {acc:.5f}")
        
        # Simple weighted average for now, we can optimize weights later
        weight = 1.0 / len(models)
        oof_preds += curr_oof * weight
        test_preds += curr_test * weight
        
    final_acc = accuracy_score(y, np.argmax(oof_preds, axis=1))
    print(f"Ensemble CV Accuracy: {final_acc:.5f}")
    
    print("Generating submission...")
    final_preds = np.argmax(test_preds, axis=1)
    final_labels = target_encoder.inverse_transform(final_preds)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'Target': final_labels
    })
    
    submission.to_csv("submission_improved.csv", index=False)
    print("Done! Saved as submission_improved.csv")

if __name__ == "__main__":
    main()
