from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.stats import mstats
import joblib
import os

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Transform temporal features into cyclical features using sine and cosine"""
    def __init__(self, features=['Departure_Month', 'Departure_Day', 'Departure_Year']):
        self.features = features
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for col in self.features:
            X[f'{col}_sin'] = np.sin(2 * np.pi * X[col]/X[col].max())
            X[f'{col}_cos'] = np.cos(2 * np.pi * X[col]/X[col].max())
            # Remove original column after encoding
            X = X.drop(col, axis=1)
        return X

class AirportEncoder(BaseEstimator, TransformerMixin):
    """Encode airport codes using LabelEncoder"""
    def __init__(self):
        self.start_encoder = LabelEncoder()
        self.dest_encoder = LabelEncoder()
        
    def fit(self, X, y=None):
        self.start_encoder.fit(X['startingAirport'])
        self.dest_encoder.fit(X['destinationAirport'])
        return self
        
    def transform(self, X):
        X = X.copy()
        X['startingAirport_encoded'] = self.start_encoder.transform(X['startingAirport'])
        X['destinationAirport_encoded'] = self.dest_encoder.transform(X['destinationAirport'])
        # Remove original columns after encoding
        X = X.drop(['startingAirport', 'destinationAirport'], axis=1)
        return X

class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    """Winsorize numerical values to handle outliers"""
    def __init__(self, column='totalFare', limits=[0.05, 0.05]):
        self.column = column
        self.limits = limits
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X[self.column] = mstats.winsorize(X[self.column].values, limits=self.limits)
        return X

def create_pipeline(model):
    """Create the complete preprocessing and model pipeline"""
    
    # Create preprocessing steps
    preprocessor = Pipeline([
        ('airport_encoder', AirportEncoder()),
        ('cyclical_encoder', CyclicalEncoder()),
        ('categorical', ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             ['depatureTimeCategory', 'Cabin_Type'])
        ], remainder='passthrough'))
    ])
    
    # Create full pipeline
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def train_model(model, X_train, y_train, model_name='flight_price_model', output_path='models/'):
    """
    Train a model using the preprocessing pipeline
    
    Parameters:
    -----------
    model : estimator object
        The machine learning model to be used
    X_train : pandas DataFrame
        Training features (already cleaned using clean_data function)
    y_train : pandas Series
        Training target variable
    model_name : str
        Name to use when saving the model
    output_path : str
        Directory where to save the model
        
    Returns:
    --------
    pipeline : Pipeline
        The fitted pipeline
    """
    # Create the pipeline
    pipeline = create_pipeline(model)
    
    # Print initial data information
    print("Training Data Shape:", X_train.shape)
    print("\nFeature Names:", X_train.columns.tolist())
    print("\nSample of training data:")
    print(X_train.head())
    
    # Fit the pipeline
    print("\nFitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Save the pipeline
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f"{model_name}.joblib")
    joblib.dump(pipeline, save_path)
    print(f"\nPipeline saved as: {save_path}")
    
    return pipeline
