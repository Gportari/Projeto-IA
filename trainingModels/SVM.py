import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KerasSVM:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, learning_rate=0.01, epochs=100, batch_size=32):
        """
        Initialize Keras-based SVM model using a neural network
        
        Parameters:
        -----------
        C : float
            Regularization parameter (inverse of regularization strength)
        kernel : str
            Kernel type to use ('linear', 'rbf', 'poly')
        gamma : str or float
            Kernel coefficient for 'rbf' and 'poly'
        degree : int
            Degree for 'poly' kernel
        learning_rate : float
            Learning rate for the optimizer
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.feature_mean = None
        self.feature_std = None
    
    def build_model(self, input_shape):
        """
        Build the Keras model for SVM-like behavior
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input features
        """
        # Determine regularization based on C parameter (inverse relationship)
        reg_strength = 1.0 / (self.C + 1e-10)
        
        # Build model architecture based on kernel type
        inputs = keras.Input(shape=input_shape)
        
        if self.kernel == 'linear':
            # Linear kernel - simpler architecture
            x = layers.Dense(64, activation='relu', 
                            kernel_regularizer=regularizers.l2(reg_strength))(inputs)
            x = layers.Dense(32, activation='relu',
                            kernel_regularizer=regularizers.l2(reg_strength))(x)
        
        elif self.kernel == 'rbf':
            # RBF kernel - deeper architecture with more units
            x = layers.Dense(128, activation='relu',
                            kernel_regularizer=regularizers.l2(reg_strength))(inputs)
            x = layers.Dense(64, activation='relu',
                            kernel_regularizer=regularizers.l2(reg_strength))(x)
            x = layers.Dense(32, activation='relu',
                            kernel_regularizer=regularizers.l2(reg_strength))(x)
        
        elif self.kernel == 'poly':
            # Polynomial kernel - use degree to determine architecture
            units = 32 * self.degree
            x = layers.Dense(units, activation='relu',
                            kernel_regularizer=regularizers.l2(reg_strength))(inputs)
            x = layers.Dense(units // 2, activation='relu',
                            kernel_regularizer=regularizers.l2(reg_strength))(x)
            x = layers.Dense(units // 4, activation='relu',
                            kernel_regularizer=regularizers.l2(reg_strength))(x)
        
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")
        
        # Output layer for binary classification
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit(self, X, y, validation_split=0.2, callbacks=None):
        """
        Train the SVM-like model
        
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
        # Normalize features
        X_normalized, self.feature_mean, self.feature_std = self._normalize_features(X)
        
        # Convert labels to expected format for binary classification
        y_formatted = np.array(y)
        # Convert -1 to 0 if present (common in SVM datasets)
        if np.any(y_formatted == -1):
            y_formatted = np.where(y_formatted == -1, 0, y_formatted)
        
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
    
    def _normalize_features(self, X, is_training=True):
        """
        Normalize features using mean and standard deviation
        
        Parameters:
        -----------
        X : array-like
            Features to normalize
        is_training : bool
            Whether this is during training (to compute stats) or prediction
        
        Returns:
        --------
        X_normalized : array
            Normalized features
        """
        X_array = np.array(X)
        
        if is_training:
            # Compute mean and std during training
            self.feature_mean = np.mean(X_array, axis=0)
            self.feature_std = np.std(X_array, axis=0)
            # Avoid division by zero
            self.feature_std = np.where(self.feature_std == 0, 1e-7, self.feature_std)
        
        # Normalize using stored stats
        X_normalized = (X_array - self.feature_mean) / self.feature_std
        
        return X_normalized, self.feature_mean, self.feature_std if is_training else None
    
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
        X_normalized, _, _ = self._normalize_features(X, is_training=False)
        
        # Get probability predictions
        y_prob = self.model.predict(X_normalized, verbose=0)
        
        # Convert to binary predictions (0 or 1)
        y_pred = (y_prob > 0.5).astype(int)
        
        # Convert 0 to -1 to match original SVM format
        y_pred = np.where(y_pred == 0, -1, 1)
        
        return y_pred.flatten()

    def predict_proba(self, X):
        """
        Predict class probabilities using the sigmoid output.
        Returns array of shape (n_samples, 2): [P(-1), P(1)].
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        X_normalized = (np.array(X, dtype=float) - self.feature_mean) / self.feature_std
        y_prob = self.model.predict(X_normalized, verbose=0)
        return np.hstack([1 - y_prob, y_prob])
    
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
        
        # Normalize features using stored stats
        X_normalized, _, _ = self._normalize_features(X, is_training=False)
        
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
        # Get predictions
        y_pred = self.predict(X)
        
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

        # Include a small sample of raw probabilities
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
    Create a Keras SVM model from configuration
    
    Parameters:
    -----------
    config : dict
        Model configuration parameters
    training_params : dict, optional
        Additional training parameters
    
    Returns:
    --------
    model : KerasSVM
        Configured model instance
    """
    # Set default training parameters
    if training_params is None:
        training_params = {}
    
    # Extract parameters from config
    C = float(config.get('svm_C', 1.0))
    kernel = config.get('svm_kernel', 'rbf')
    gamma = config.get('svm_gamma', 'scale')
    degree = int(config.get('svm_degree', 3))
    learning_rate = float(training_params.get('learning_rate', 0.01))
    epochs = int(training_params.get('epochs', 100))
    batch_size = int(training_params.get('batch_size', 32))
    
    # Create and return model
    return KerasSVM(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

def train_and_evaluate(model_config, X_train, X_test, y_train, y_test, training_params=None):
    """
    Train and evaluate a Keras SVM model
    
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
        'name': model_config.get('name', 'SVM'),
        **metrics
    }
    
    return results, model