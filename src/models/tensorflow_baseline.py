import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def build_and_train_tf_baseline(data_path, model_save_path, scaler_save_path):
    print("Loading data for TensorFlow Baseline...")
    df = pd.read_parquet(data_path)
    
    feature_cols = [
        'price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
        'demand_score_7d', 'demand_score_30d', 'demand_velocity',
        'inventory_ratio', 'price_percentile_in_category',
        'competitor_delta', 'review_elasticity'
    ]
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    df = df.dropna(subset=['optimal_price'])
    
    X = df[feature_cols].values
    y = df['optimal_price'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (Deep Learning requires scaled inputs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build Keras Sequential Architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(len(feature_cols),)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear') # Regression output
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    # Early stopping prevents overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training TensorFlow Baseline Model...")
    history = model.fit(X_train_scaled, y_train, 
                        validation_split=0.2,
                        epochs=50, 
                        batch_size=256,
                        callbacks=[early_stop],
                        verbose=1)
    
    # Evaluate Accuracy
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test MAE: {mae:.2f}")
    
    # Save the architecture and scaler
    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"Model successfully saved to {model_save_path}")
    print(f"Scaler successfully saved to {scaler_save_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_file = os.path.join(project_root, 'data', 'processed', 'features.parquet')
    model_file = os.path.join(project_root, 'models', 'tf_pricing_baseline.h5')
    scaler_file = os.path.join(project_root, 'models', 'tf_scaler.pkl')
    
    build_and_train_tf_baseline(data_file, model_file, scaler_file)
