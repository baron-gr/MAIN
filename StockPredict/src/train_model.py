# imports
import os
from typing import Optional, Callable

from datetime import datetime
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import pandas as pd
import pickle
from argparse import ArgumentParser

# project imports
from src.preprocess import transform_features_targets
from src.features_pipepline import preprocess_pipeline
from src.hyperparams import best_hyperparams

from src.paths import MODELS_DIR
from src.logger import get_console_logger

# log run
logger = get_console_logger()

# function to pass model name
def choose_model(model: str) -> Callable:
    """
    Returns model function given the model name
    """
    if model == 'lasso':
        return Lasso
    elif model == 'light':
        return LGBMRegressor
    elif model == 'boost':
        return XGBRegressor
    elif model == 'forest':
        return RandomForestRegressor
    else:
        return ValueError(f'Unknown model name: {model}')

# function to train model
def train(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    tune_hyperparams: Optional[bool] = False,
    hyperparam_trials: Optional[int] = 10
) -> None:
    """
    Train boosting tree model using input features X and target y
    """
    
    # pull desired model
    model_func = choose_model(model)
    
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
    
    if not tune_hyperparams:
        # create full pipeline with default hyperparams
        logger.info('Using default hyperparameters')
        
        pipeline = make_pipeline(
            preprocess_pipeline(),
            model_func()
        )
    else:
        # find best hyperparams using cross-validation
        logger.info('Finding best hyperparameters using cross-validation')
        
        preprocess_hyperparams, model_hyperparams = best_hyperparams(model_func, hyperparam_trials, X_train, y_train)
        logger.info(f'Best preprocessing hyperparameters: {preprocess_hyperparams}')
        logger.info(f'Best model hyperparameters: {model_hyperparams}')
        
        pipeline = make_pipeline(
            preprocess_pipeline(**preprocess_hyperparams),
            model_func(**model_hyperparams)
        )
    
    # train model
    logger.info('Fitting model')
    pipeline.fit(X_train, y_train)
    
    # compute mae
    predictions = pipeline.predict(X_test)
    mae_error = mean_absolute_error(y_test, predictions)
    logger.info(f'Test MAE: {mae_error}')
    
    # save model name and timestamp
    timestamp = datetime.today().strftime("%Y-%m-%d")
    logger.info('Saving model to disk')
    with open(MODELS_DIR / f'{model}_{timestamp}.pkl', "wb") as f:
        pickle.dump(pipeline, f)
    
if __name__ == '__main__':
    # create arguments for CLI execution
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='lasso')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    
    # create features and target dataframes
    logger.info('Generating features and targets')
    features_df, target_df = transform_features_targets()
    
    # reduce input size to speed training
    if args.sample is not None:
        features_df = features_df.head(args.sample)
        target_df = target_df.head(args.sample)
        
    # train model
    logger.info('Training model')
    train(features_df, target_df,
          model=args.model,
          tune_hyperparams=args.tune,
          hyperparam_trials=args.trials
          )