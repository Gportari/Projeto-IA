import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KerasLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=32, 
                 l1_regularization=0.0, l2_regularization=0.0):
        """
        Initialize Keras-based Logistic Regression model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for the optimizer
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        l1_regularization : float
            L1 regularization strength (similar to 'l1' penalty in sklearn)
        l2_regularization : float
            L2 regularization strength (similar to 'l2' penalty in sklearn)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.model = None
        self.history = None
        # Feature normalization parameters
        self.feature_mean = None
        self.feature_std = None
    
    def build_model(self, input_shape):
        """
        Build the Keras model for logistic regression
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input features
        """
        # Configure regularization
        regularizer = None
        if self.l1_regularization > 0 and self.l2_regularization > 0:
            regularizer = keras.regularizers.L1L2(l1=self.l1_regularization, l2=self.l2_regularization)
        elif self.l1_regularization > 0:
            regularizer = keras.regularizers.L1(l1=self.l1_regularization)
        elif self.l2_regularization > 0:
            regularizer = keras.regularizers.L2(l2=self.l2_regularization)
        
        # Build model
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    def _normalize_features(self, X):
        """
        Normalize features using standardization (z-score).

        Returns normalized features and sets mean/std for later inference.
        """
        X_arr = np.array(X, dtype=float)
        if self.feature_mean is None or self.feature_std is None:
            self.feature_mean = X_arr.mean(axis=0)
            self.feature_std = X_arr.std(axis=0)
            self.feature_std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        X_norm = (X_arr - self.feature_mean) / self.feature_std
        return X_norm
    
    def fit(self, X, y, validation_split=0.2, callbacks=None):
        """
        Train the logistic regression model
        
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
        
        # Normalize features
        X_normalized = self._normalize_features(X)

        # Build model if not already built
        if self.model is None:
            self.build_model(X_normalized.shape[1:])
        
        # Train model
        self.history = self.model.fit(
            X_normalized, y_formatted,
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
        
        # Normalize features using stored stats
        X_normalized = (np.array(X, dtype=float) - self.feature_mean) / self.feature_std
        # Get probability predictions
        y_prob = self.model.predict(X_normalized, verbose=0)
        
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
        
        # Normalize features
        X_normalized = (np.array(X, dtype=float) - self.feature_mean) / self.feature_std
        # Get probability predictions
        y_prob = self.model.predict(X_normalized, verbose=0)
        
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
        # Convert labels if needed
        y_formatted = np.array(y)
        if np.any(y_formatted == -1):
            y_formatted = np.where(y_formatted == -1, 0, y_formatted)
        
        # Get predictions
        y_pred = self.predict(X)
        y_pred_formatted = np.where(y_pred == -1, 0, y_pred)
        
        # Calculate metrics
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, y_pred, labels=[-1, 1])

        # Training history diagnostics
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
    Create a Keras Logistic Regression model from configuration
    
    Parameters:
    -----------
    config : dict
        Model configuration parameters
    training_params : dict, optional
        Additional training parameters
    
    Returns:
    --------
    model : KerasLogisticRegression
        Configured model instance
    """
    # Set default training parameters
    if training_params is None:
        training_params = {}
    
    # Extract parameters from config
    learning_rate = float(training_params.get('learning_rate', 0.01))
    epochs = int(training_params.get('epochs', 100))
    batch_size = int(training_params.get('batch_size', 32))
    
    # Map regularization parameters
    penalty = config.get('logreg_penalty', 'l2')
    C = float(config.get('logreg_C', 1.0))
    
    # Convert C to regularization strength (inverse relationship)
    reg_strength = 1.0 / C if C > 0 else 0.0
    
    # Configure L1 and L2 regularization based on penalty
    l1_reg = reg_strength if penalty == 'l1' else 0.0
    l2_reg = reg_strength if penalty == 'l2' else 0.0
    
    # Create and return model
    return KerasLogisticRegression(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        l1_regularization=l1_reg,
        l2_regularization=l2_reg
    )

def train_and_evaluate(model_config, X_train, X_test, y_train, y_test, training_params=None):
    """
    Train and evaluate a Keras Logistic Regression model
    
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
        'name': model_config.get('name', 'Logistic Regression'),
        **metrics
    }
    
    return results, model