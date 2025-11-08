import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KerasKNN:
    def __init__(self, n_neighbors=5, learning_rate=0.01, epochs=100, batch_size=32):
        """
        Initialize Keras-based KNN-like model using a neural network
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors (used to determine network complexity)
        learning_rate : float
            Learning rate for the optimizer
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        self.n_neighbors = n_neighbors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """
        Build the Keras model for KNN-like behavior
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input features
        """
        # Determine network complexity based on n_neighbors
        units = max(16, self.n_neighbors * 4)
        
        # Build model with multiple layers to approximate KNN behavior
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(units, activation='relu'),
            layers.Dense(units // 2, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit(self, X, y, validation_split=0.2, callbacks=None):
        """
        Train the KNN-like model
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Target values
        validation_split : float
            Fraction of data to use for validation
        
        Returns:
        --------
        self : object
            Returns self
        """
        # Convert labels to expected format for binary classification
        y_formatted = np.array(y)
        # Convert -1 to 0 if present (common in some datasets)
        if np.any(y_formatted == -1):
            y_formatted = np.where(y_formatted == -1, 0, y_formatted)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X.shape[1:])
        
        # Train model
        self.history = self.model.fit(
            X, y_formatted,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=callbacks or []
        )
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like
            Features to predict
        
        Returns:
        --------
        y_pred : array
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Get probability predictions
        y_prob = self.model.predict(X, verbose=0)
        
        # Convert to binary predictions (0 or 1)
        y_pred = (y_prob > 0.5).astype(int)
        
        # Convert 0 to -1 to match original format if needed
        y_pred = np.where(y_pred == 0, -1, 1)
        
        return y_pred.flatten()
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        
        Parameters:
        -----------
        X : array-like
            Features to predict
        
        Returns:
        --------
        y_prob : array
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Get probability predictions
        y_prob = self.model.predict(X, verbose=0)
        
        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.hstack([1-y_prob, y_prob])
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            True labels
        
        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        # Get predictions
        y_pred = self.predict(X)

        # Confusion matrix in original label space [-1, 1]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, y_pred, labels=[-1, 1])

        # Training history diagnostics if available
        history_dict = None
        if self.history is not None and hasattr(self.history, 'history'):
            h = self.history.history
            history_dict = {
                'loss': list(map(float, h.get('loss', []))),
                'val_loss': list(map(float, h.get('val_loss', []))),
                'accuracy': list(map(float, h.get('accuracy', []))),
                'val_accuracy': list(map(float, h.get('val_accuracy', [])))
            }

        # Include a small sample of raw probabilities for inspection
        try:
            proba_sample = self.predict_proba(X[:10]).tolist()
        except Exception:
            proba_sample = None

        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y, y_pred, zero_division=0)),
            'confusion_matrix': cm.tolist(),
            'history': history_dict,
            'probabilities_sample': proba_sample
        }

        return metrics

def create_model_from_config(config, training_params=None):
    """
    Create a Keras KNN model from configuration
    
    Parameters:
    -----------
    config : dict
        Model configuration parameters
    training_params : dict, optional
        Additional training parameters
    
    Returns:
    --------
    model : KerasKNN
        Configured model instance
    """
    # Set default training parameters
    if training_params is None:
        training_params = {}
    
    # Extract parameters from config
    n_neighbors = int(config.get('knn_n_neighbors', 5))
    learning_rate = float(training_params.get('learning_rate', 0.01))
    epochs = int(training_params.get('epochs', 100))
    batch_size = int(training_params.get('batch_size', 32))
    
    # Create and return model
    return KerasKNN(
        n_neighbors=n_neighbors,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

def train_and_evaluate(model_config, X_train, X_test, y_train, y_test, training_params=None):
    """
    Train and evaluate a Keras KNN model
    
    Parameters:
    -----------
    model_config : dict
        Model configuration parameters
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    training_params : dict, optional
        Additional training parameters
    
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics
    """
    # Create model
    model = create_model_from_config(model_config, training_params)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Add model name to results
    results = {
        'name': model_config.get('name', 'KNN'),
        **metrics
    }
    
    return results, model