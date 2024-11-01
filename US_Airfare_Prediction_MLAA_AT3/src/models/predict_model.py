import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def predict_model(X_test, y_test, pipeline):
    """
    Load model, make predictions and return evaluation metrics.
    
    Parameters:
    -----------
    X_test : pandas DataFrame
        Test features
    y_test : pandas Series
        True values
    model_path : str
        Path to saved model pipeline
        
    Returns:
    --------
    dict: Dictionary containing predictions and evaluation metrics
    """
    try:
        # Load the pipeline and make predictions
        # pipeline = joblib.load(model_path)
        predictions = pipeline.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        # Print metrics
        print("\nModel Performance:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        print(f"R2 Score: {r2:.3f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {
            'predictions': predictions,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
        }
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise