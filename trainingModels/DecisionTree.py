import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KerasDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, learning_rate=0.01, epochs=100, batch_size=32):
        """
        Initialize Keras-based Decision Tree model using a neural network
        
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree (controls network complexity)
        min_samples_split : int
            Minimum samples required to split (controls network width)
        learning_rate : float
            Learning rate for the optimizer
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """
        Build the Keras model for Decision Tree-like behavior
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input features
        """
        # Calculate network complexity based on tree parameters
        # More depth = more layers, more min_samples_split = fewer neurons
        n_layers = min(10, max(3, self.max_depth))
        width_factor = max(1, 32 // self.min_samples_split)
        base_units = 32 * width_factor
        
        # Build model with multiple layers
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        
        # Add layers with decreasing width to mimic tree structure
        for i in range(n_layers):
            units = base_units // (2 ** min(i, 3))  # Decrease units by power of 2
            model.add(layers.Dense(
                units=max(8, units),
                activation='relu',
                kernel_initializer='he_normal'
            ))
            # Add dropout for regularization
            if i < n_layers - 1:  # No dropout on final hidden layer
                model.add(layers.Dropout(0.2))
        
        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit(self, X, y, validation_split=0.2):
        """
        Train the Decision Tree-like model
        
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
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        self.history = self.model.fit(
            X, y_formatted,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
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
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y, y_pred, zero_division=0))
        }
        
        return metrics

def create_model_from_config(config, training_params=None):
    """
    Create a Keras Decision Tree model from configuration
    
    Parameters:
    -----------
    config : dict
        Model configuration parameters
    training_params : dict, optional
        Additional training parameters
    
    Returns:
    --------
    model : KerasDecisionTree
        Configured model instance
    """
    # Set default training parameters
    if training_params is None:
        training_params = {}
    
    # Extract parameters from config
    max_depth = int(config.get('tree_max_depth', 5))
    min_samples_split = int(config.get('tree_min_samples_split', 2))
    learning_rate = float(training_params.get('learning_rate', 0.01))
    epochs = int(training_params.get('epochs', 100))
    batch_size = int(training_params.get('batch_size', 32))
    
    # Create and return model
    return KerasDecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

def train_and_evaluate(model_config, X_train, X_test, y_train, y_test, training_params=None):
    """
    Train and evaluate a Keras Decision Tree model
    
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
        'name': model_config.get('name', 'Decision Tree'),
        **metrics
    }
    
    return results, model