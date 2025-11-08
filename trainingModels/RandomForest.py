import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class KerasRandomForest:
    def __init__(self, n_estimators=10, max_depth=5, learning_rate=0.01, epochs=100, batch_size=32):
        """
        Initialize Keras-based Random Forest model using ensemble of neural networks
        
        Parameters:
        -----------
        n_estimators : int
            Number of estimators (sub-networks) in the ensemble
        max_depth : int
            Maximum depth of each tree (controls network complexity)
        learning_rate : float
            Learning rate for the optimizer
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = []
        self.histories = []
        self.feature_indices = []
    
    def _build_single_model(self, input_shape, n_features_subset):
        """
        Build a single neural network for the ensemble
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input features
        n_features_subset : int
            Number of features to use for this model
            
        Returns:
        --------
        model : keras.Model
            A single neural network model
        """
        # Calculate network complexity based on tree parameters
        n_layers = min(8, max(3, self.max_depth))
        base_units = 32
        
        # Build model with multiple layers
        inputs = keras.Input(shape=(n_features_subset,))
        x = inputs
        
        # Add layers with decreasing width to mimic tree structure
        for i in range(n_layers):
            units = base_units // (2 ** min(i, 3))  # Decrease units by power of 2
            x = layers.Dense(
                units=max(8, units),
                activation='relu',
                kernel_initializer='he_normal'
            )(x)
            # Add dropout for regularization
            if i < n_layers - 1:  # No dropout on final hidden layer
                x = layers.Dropout(0.2)(x)
        
        # Output layer for binary classification
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create and compile model
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y, validation_split=0.2, callbacks=None):
        """
        Train the Random Forest-like model
        
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
        X_array = np.array(X)
        n_features = X_array.shape[1]
        
        # Convert labels to expected format for binary classification
        y_formatted = np.array(y)
        # Convert -1 to 0 if present (common in some datasets)
        if np.any(y_formatted == -1):
            y_formatted = np.where(y_formatted == -1, 0, y_formatted)
        
        # Clear any existing models
        self.models = []
        self.histories = []
        self.feature_indices = []
        
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train multiple models (estimators)
        for i in range(self.n_estimators):
            # Random feature selection (like in Random Forest)
            n_features_subset = max(1, int(np.sqrt(n_features)))
            feature_indices = np.random.choice(
                n_features, size=n_features_subset, replace=False
            )
            self.feature_indices.append(feature_indices)
            
            # Extract subset of features
            X_subset = X_array[:, feature_indices]
            
            # Bootstrap sampling (with replacement)
            n_samples = X_array.shape[0]
            sample_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            X_bootstrap = X_subset[sample_indices]
            y_bootstrap = y_formatted[sample_indices]
            
            # Build and train model
            model = self._build_single_model(X_array.shape[1:], n_features_subset)
            history = model.fit(
                X_bootstrap, y_bootstrap,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping] + (callbacks or []),
                verbose=0
            )
            
            # Store model and history
            self.models.append(model)
            self.histories.append(history)
        
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
        if not self.models:
            raise ValueError("Model has not been trained yet.")
        
        X_array = np.array(X)
        n_samples = X_array.shape[0]
        
        # Get predictions from all models
        all_predictions = np.zeros((n_samples, len(self.models)))
        
        for i, (model, feature_indices) in enumerate(zip(self.models, self.feature_indices)):
            # Extract subset of features
            X_subset = X_array[:, feature_indices]
            
            # Get probability predictions
            y_prob = model.predict(X_subset, verbose=0).flatten()
            
            # Convert to binary predictions (0 or 1)
            all_predictions[:, i] = (y_prob > 0.5).astype(int)
        
        # Majority voting
        y_pred = np.mean(all_predictions, axis=1) >= 0.5
        
        # Convert to -1/1 format
        y_pred = np.where(y_pred, 1, -1)
        
        return y_pred
    
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
        if not self.models:
            raise ValueError("Model has not been trained yet.")
        
        X_array = np.array(X)
        n_samples = X_array.shape[0]
        
        # Get probability predictions from all models
        all_probs = np.zeros((n_samples, len(self.models)))
        
        for i, (model, feature_indices) in enumerate(zip(self.models, self.feature_indices)):
            # Extract subset of features
            X_subset = X_array[:, feature_indices]
            
            # Get probability predictions
            all_probs[:, i] = model.predict(X_subset, verbose=0).flatten()
        
        # Average probabilities
        avg_prob = np.mean(all_probs, axis=1)
        
        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.column_stack([1 - avg_prob, avg_prob])
    
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
    Create a Keras Random Forest model from configuration
    
    Parameters:
    -----------
    config : dict
        Model configuration parameters
    training_params : dict, optional
        Additional training parameters
    
    Returns:
    --------
    model : KerasRandomForest
        Configured model instance
    """
    # Set default training parameters
    if training_params is None:
        training_params = {}
    
    # Extract parameters from config
    n_estimators = int(config.get('rf_n_estimators', 10))
    max_depth = int(config.get('rf_max_depth', 5))
    learning_rate = float(training_params.get('learning_rate', 0.01))
    epochs = int(training_params.get('epochs', 100))
    batch_size = int(training_params.get('batch_size', 32))
    
    # Create and return model
    return KerasRandomForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )

def train_and_evaluate(model_config, X_train, X_test, y_train, y_test, training_params=None):
    """
    Train and evaluate a Keras Random Forest model
    
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
        'name': model_config.get('name', 'Random Forest'),
        **metrics
    }
    
    return results, model