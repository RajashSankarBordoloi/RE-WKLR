from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

def sampling_preprocessing_pipeline(X, y, dataset_name="general", test_size=0.2):
    """Complete sampling and preprocessing pipeline based on the paper's scheme."""
    
    # Ensure X and y are DataFrames
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        y = pd.Series(y)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  # Take the first column if y is a DataFrame
    
    # Normalize X (mean 0, std 1)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Convert categorical y to binary (0 = non-event, 1 = event)
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.values), name='target')
    else:
        y = pd.Series(y.values, name='target')
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split classes for sampling
    class_0_train = X_train[y_train == 0]
    class_1_train = X_train[y_train == 1]
    # y_0_train = y_train[y_train == 0]
    # y_1_train = y_train[y_train == 1]
    
    # Balanced and imbalanced training sets
    if dataset_name == "spam":
        X_train_bal = pd.concat([
            class_0_train.sample(200, random_state=42),
            class_1_train.sample(200, random_state=42)
        ])
        y_train_bal = y.loc[X_train_bal.index]
        
        X_train_imb = pd.concat([
            class_0_train.sample(200, random_state=42),
            class_1_train.sample(100, random_state=42)
        ])
        y_train_imb = y.loc[X_train_imb.index]
    else:
        X_train_bal = pd.concat([
            class_0_train.sample(40, random_state=42),
            class_1_train.sample(40, random_state=42)
        ])
        y_train_bal = y.loc[X_train_bal.index]
        
        X_train_imb = pd.concat([
            class_0_train.sample(40, random_state=42),
            class_1_train.sample(15, random_state=42)
        ])
        y_train_imb = y.loc[X_train_imb.index]
    
    # Test set rarity — 5% events to non-events ratio (8% for SPECT Heart)
    class_0_test = X_test[y_test == 0]
    class_1_test = X_test[y_test == 1]
    y_0_test = y_test[y_test == 0]
    y_1_test = y_test[y_test == 1]
    
    test_ratio = 0.05 if dataset_name != "SPECT Heart" else 0.08
    n_test_events = int(round(len(class_0_test) * test_ratio))
    
    sampled_class_1 = class_1_test.sample(n_test_events, random_state=42)
    sampled_y_1 = y_1_test.loc[sampled_class_1.index]
    
    X_test_final = pd.concat([class_0_test, sampled_class_1])
    y_test_final = pd.concat([y_0_test, sampled_y_1])
    
    # Shuffle X_test_final and y_test_final together
    test_final = pd.concat([X_test_final, y_test_final], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)
    X_test_final = test_final.drop(columns=["target"])
    y_test_final = test_final["target"]
    
    return (X_train_bal, y_train_bal), (X_train_imb, y_train_imb), (X_test_final, y_test_final)
