import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import mstats
import pickle
import joblib
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from datetime import datetime
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

def tune_pipeline(pipeline, param_grid, X_train, y_train, 
                 search_type='random',
                 n_iter=100,
                 cv=5,
                 scoring='neg_root_mean_squared_error',
                 n_jobs=-1,
                 verbose=2):
    """
    Tune pipeline hyperparameters and save the best pipeline.
    
    Parameters:
    -----------
    pipeline : sklearn Pipeline
        The pipeline to tune
    param_grid : dict
        Dictionary with parameter names as keys. Parameter names should be in format:
        'step_name__parameter_name'
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    search_type : str, default='random'
        Type of search ('random' or 'grid')
    n_iter : int, default=100
        Number of iterations for RandomizedSearchCV
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='neg_root_mean_squared_error'
        Scoring metric to use
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : int, default=2
        Verbosity level
        
    Returns:
    --------
    dict
        Dictionary containing best pipeline, best parameters, and cv results
    """
    try:
        # Create the appropriate search CV
        if search_type.lower() == 'random':
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=42
            )
        else:  # grid search
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose
            )
        
        # Perform the search
        print(f"\nStarting {search_type} search CV...")
        search.fit(X_train, y_train)
        
        # Print best parameters and score
        print("\nBest Parameters:")
        print("----------------")
        for param, value in search.best_params_.items():
            print(f"{param}: {value}")
            
        print("\nBest Score:")
        print("-----------")
        print(f"CV Score: {abs(search.best_score_):.4f}")  # Taking abs as scoring might be negative
        
        # Create models directory if it doesn't exist
        # os.makedirs('models', exist_ok=True)
        
        # Save the best pipeline
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'../models/best_pipeline_{search_type}_{timestamp}.joblib'
        joblib.dump(search.best_estimator_, model_filename)
        print(f"\nBest pipeline saved as: {model_filename}")
        
        # Save the CV results
        cv_results = pd.DataFrame(search.cv_results_)
        results_filename = f'../models/cv_results_{search_type}_{timestamp}.csv'
        cv_results.to_csv(results_filename, index=False)
        print(f"CV results saved as: {results_filename}")
        
        return {
            'best_pipeline': search.best_estimator_,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': cv_results
        }
        
    except Exception as e:
        print(f"Error during pipeline tuning: {str(e)}")
        raise