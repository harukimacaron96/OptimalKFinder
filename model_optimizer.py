from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_optimal_k_with_distance(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    param_grid = {
        'n_neighbors': range(1, 50),
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    randomized_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, n_iter=50, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
    randomized_search.fit(X, y)
    return randomized_search.best_params_

def save_model(model: Any, filename: str) -> None:
    """Save the trained model to a file."""
    try:
        joblib.dump(model, filename)
        logging.info(f"Model saved: {filename}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")

def load_model(filename: str) -> Any:
    """Load a model from a file."""
    try:
        model = joblib.load(filename)
        logging.info(f"Loaded existing model: {filename}")
        return model
    except FileNotFoundError:
        logging.warning(f"Model file not found: {filename}")
        return None
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

def save_scaler(scaler: Any, filename: str) -> None:
    """Save the scaler to a file."""
    try:
        joblib.dump(scaler, filename)
        logging.info(f"Scaler saved: {filename}")
    except Exception as e:
        logging.error(f"Failed to save scaler: {e}")

def load_scaler(filename: str) -> Any:
    """Load a scaler from a file."""
    try:
        scaler = joblib.load(filename)
        logging.info(f"Loaded existing scaler: {filename}")
        return scaler
    except FileNotFoundError:
        logging.warning(f"Scaler file not found: {filename}")
        return None
    except Exception as e:
        logging.error(f"Failed to load scaler: {e}")
        return None

def scale_data(X: np.ndarray, scaler_filename: str = 'scaler.pkl') -> np.ndarray:
    """Scale the data using StandardScaler and save the scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    save_scaler(scaler, scaler_filename)
    return X_scaled

def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate the model using cross-validation."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    logging.info(f"Ensemble model mean accuracy: {mean_score:.4f}")
    return mean_score

# Scale the data
X_scaled = scale_data(X)

# Find the optimal k and distance metric
best_params = find_optimal_k_with_distance(X_scaled, y)
optimal_k = best_params['n_neighbors']
best_metric = best_params['metric']
best_weights = best_params['weights']
logging.info(f"Optimal k: {optimal_k}, Optimal metric: {best_metric}, Optimal weights: {best_weights}")

# Create the k-NN model
knn_best = KNeighborsClassifier(n_neighbors=optimal_k, weights=best_weights, metric=best_metric)

# Create the Random Forest model
random_forest = RandomForestClassifier(random_state=42)

# Set up the ensemble learning
ensemble_model = VotingClassifier(estimators=[
    ('knn', knn_best),
    ('rf', random_forest)
], voting='soft')

# Evaluate the ensemble model
evaluate_model(ensemble_model, X_scaled, y)

# Train the model
ensemble_model.fit(X_scaled, y)

# Save the trained model
save_model(ensemble_model, 'ensemble_model.pkl')

# Predict new data points
if 'new_data' in locals():
    scaler = load_scaler('scaler.pkl')
    if scaler is not None and ensemble_model is not None:
        try:
            new_data_scaled = scaler.transform(new_data)
            prediction = ensemble_model.predict(new_data_scaled)
            logging.info(f"Prediction for new data points: {prediction}")
        except Exception as e:
            logging.error(f"Failed to predict: {e}")
