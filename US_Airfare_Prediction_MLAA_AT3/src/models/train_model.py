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
import pandas as pd
from datetime import datetime
import os
class SeparateAirportEncoder(BaseEstimator, TransformerMixin):
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
        return X

class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col + '_sin'] = np.sin(2 * np.pi * X[col]/X[col].max())
            X[col + '_cos'] = np.cos(2 * np.pi * X[col]/X[col].max())
        return X
    
class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=[0.05, 0.05]):
        self.limits = limits
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_df = pd.DataFrame(X)
        winsorized_data = mstats.winsorize(X_df.values, limits=self.limits, axis=0)
        return pd.DataFrame(winsorized_data, columns=X_df.columns, index=X_df.index)
    
def train_model_with_cyclical(model, X_train, y_train, output_path='models/'):
    """
    Create and train a pipeline including cyclical encoding for temporal features.
    Uses separate encoders for starting and destination airports.
    
    Parameters:
    -----------
    model : estimator object
        The machine learning model to be used
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series
        Training target variable
    output_path : str
        Path where to save the pipeline and encoders
        
    Returns:
    --------
    trained_pipeline : Pipeline
        The fitted pipeline object
    """
    
    cyclical_columns = ['Departure_Month', 'Departure_Day', 'Departure_Year']
    categorical_columns = ['depatureTimeCategory', 'Cabin_Type']
    
    preprocessing_pipeline = Pipeline([
        ('airport_encoding', SeparateAirportEncoder()),
        ('cyclical_encoding', CyclicalEncoder(cyclical_columns)),
        ('categorical_encoding', ColumnTransformer([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_columns)
        ])),
        ('winsorize', WinsorizeTransformer())
    ])
    
    full_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', model)
    ])
    
    # Fit the pipeline
    full_pipeline.fit(X_train, y_train)
    
    # Save the full pipeline
    joblib.dump(full_pipeline, f'{output_path}full_pipeline_cyclical.joblib')
    
    # Save encoders separately for convenience
    airport_encoders = {
        'start_encoder': full_pipeline.named_steps['preprocessing'].named_steps['airport_encoding'].start_encoder,
        'dest_encoder': full_pipeline.named_steps['preprocessing'].named_steps['airport_encoding'].dest_encoder
    }
    with open(f'{output_path}airport_encoders.pkl', 'wb') as f:
        pickle.dump(airport_encoders, f)
    
    return full_pipeline

def train_model(model, X_train, y_train, output_path='models/'):
    """
    Create and train a pipeline without cyclical encoding.
    Uses separate encoders for starting and destination airports.
    
    Parameters:
    -----------
    model : estimator object
        The machine learning model to be used
    X_train : pandas DataFrame
        Training features
    y_train : pandas Series
        Training target variable
    output_path : str
        Path where to save the pipeline and encoders
        
    Returns:
    --------
    trained_pipeline : Pipeline
        The fitted pipeline object
    """
    
    categorical_columns = ['depatureTimeCategory', 'Cabin_Type']
    temporal_columns = ['Departure_Month', 'Departure_Day', 'Departure_Year']
    
    preprocessing_pipeline = Pipeline([
        ('airport_encoding', SeparateAirportEncoder()),
        ('feature_encoding', ColumnTransformer([
            ('temporal', 'passthrough', temporal_columns),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_columns)
        ])),
        ('winsorize', WinsorizeTransformer())
    ])
    
    full_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', model)
    ])
    
    # Fit the pipeline
    full_pipeline.fit(X_train, y_train)
    
    # Save the full pipeline
    joblib.dump(full_pipeline, f'{output_path}full_pipeline_no_cyclical.joblib')
    
    # Save encoders separately for convenience
    airport_encoders = {
        'start_encoder': full_pipeline.named_steps['preprocessing'].named_steps['airport_encoding'].start_encoder,
        'dest_encoder': full_pipeline.named_steps['preprocessing'].named_steps['airport_encoding'].dest_encoder
    }
    with open(f'{output_path}airport_encoders.pkl', 'wb') as f:
        pickle.dump(airport_encoders, f)
    
    return full_pipeline

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