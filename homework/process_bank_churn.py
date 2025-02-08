import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

def select_columns(df: pd.DataFrame) -> List[str]:
    """
    Select input columns for the model, excluding irrelevant ones.
    """
    return list(df.columns)[3:-1]

def split_data(df: pd.DataFrame, target_col: str, input_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split raw data into training and validation sets.
    """
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    return train_df[input_cols].copy(), val_df[input_cols].copy(), train_df[target_col].copy(), val_df[target_col].copy()

def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify numeric, categorical, and binary numeric columns in the dataset.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    binary_numeric_cols = [col for col in numeric_cols if df[col].nunique() == 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_numeric_cols]
    return numeric_cols, categorical_cols, binary_numeric_cols

def scale_numeric_features(df: pd.DataFrame, numeric_cols: List[str], scaler: StandardScaler) -> pd.DataFrame:
    """
    Scale numeric features using a provided scaler.
    """
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df

def encode_categorical_features(df: pd.DataFrame, categorical_cols: List[str], encoder: OneHotEncoder) -> pd.DataFrame:
    """
    One-hot encode categorical features using a provided encoder.
    """
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    df[encoded_cols] = encoder.transform(df[categorical_cols])
    return df.drop(columns=categorical_cols)

def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> Dict[str, any]:
    """
    Preprocess raw data by selecting features, splitting data, encoding categorical variables,
    and optionally scaling numeric features.
    """
    input_cols = select_columns(raw_df)
    target_col = 'Exited'
    X_train, X_val, train_targets, val_targets = split_data(raw_df, target_col, input_cols)
    numeric_cols, categorical_cols, _ = identify_column_types(X_train)
    
    scaler = StandardScaler().fit(X_train[numeric_cols]) if scale_numeric else None
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X_train[categorical_cols])
    
    if scale_numeric:
        X_train = scale_numeric_features(X_train, numeric_cols, scaler)
        X_val = scale_numeric_features(X_val, numeric_cols, scaler)
    
    X_train = encode_categorical_features(X_train, categorical_cols, encoder)
    X_val = encode_categorical_features(X_val, categorical_cols, encoder)
    
    return {
        "X_train": X_train,
        "train_targets": train_targets,
        "X_val": X_val,
        "val_targets": val_targets,
        "input_cols": input_cols,
        "scaler": scaler,
        "encoder": encoder
    }

def preprocess_new_data(new_data: pd.DataFrame, encoder: OneHotEncoder, scaler: StandardScaler = None) -> pd.DataFrame:
    """
    Preprocess new data using previously fitted scaler and encoder.
    """
    new_data = new_data.copy()
    numeric_cols, categorical_cols, _ = identify_column_types(new_data)
    
    if scaler:
        numeric_cols = [col for col in numeric_cols if col != 'id']
        new_data = scale_numeric_features(new_data, numeric_cols, scaler)
    
    new_data = encode_categorical_features(new_data, categorical_cols, encoder)
    return new_data
