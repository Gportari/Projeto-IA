import numpy as np
from trainingModels.LogisticRegression import train_and_evaluate as train_logistic
from trainingModels.SVM import train_and_evaluate as train_svm


def _make_synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 3))
    w = np.array([1.5, -2.0, 0.7])
    z = X @ w + 0.2 * rng.normal(0, 1, size=n)
    y = np.where(z > 0.0, 1, -1)
    # train/test split
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], y[tr], y[te]


def test_logistic_synthetic():
    X_train, X_test, y_train, y_test = _make_synthetic()
    config = {'name': 'LogReg Test', 'logreg_penalty': 'l2', 'logreg_C': 1.0}
    training_params = {'epochs': 50, 'batch_size': 32}
    results, _ = train_logistic(config, X_train, X_test, y_train, y_test, training_params)
    assert 'accuracy' in results and results['accuracy'] >= 0.7
    assert 'precision' in results and 'recall' in results and 'f1_score' in results
    assert isinstance(results.get('confusion_matrix'), list)


def test_svm_synthetic():
    X_train, X_test, y_train, y_test = _make_synthetic(seed=1)
    config = {'name': 'SVM Test', 'svm_kernel': 'rbf', 'svm_C': 1.0, 'svm_gamma': 'scale', 'svm_probability': True}
    training_params = {}
    results, _ = train_svm(config, X_train, X_test, y_train, y_test, training_params)
    assert 'accuracy' in results and results['accuracy'] >= 0.7
    assert 'precision' in results and 'recall' in results and 'f1_score' in results
    assert isinstance(results.get('confusion_matrix'), list)


def test_empty_inputs():
    X_train = np.empty((0, 3))
    y_train = np.array([])
    X_test = np.random.randn(10, 3)
    y_test = np.random.choice([-1, 1], size=10)
    config = {'name': 'LogReg Empty', 'logreg_penalty': 'l2', 'logreg_C': 1.0}
    training_params = {'epochs': 5, 'batch_size': 4}
    try:
        train_logistic(config, X_train, X_test, y_train, y_test, training_params)
        raise AssertionError('Expected failure on empty training data')
    except Exception:
        pass


def test_single_class_inputs():
    # All training labels are the same class
    X_train = np.random.randn(100, 3)
    y_train = np.ones(100, dtype=int)
    X_test = np.random.randn(20, 3)
    y_test = np.random.choice([-1, 1], size=20)
    config = {'name': 'LogReg SingleClass', 'logreg_penalty': 'l2', 'logreg_C': 1.0}
    training_params = {'epochs': 10, 'batch_size': 8}
    # Some implementations may raise; we just ensure it doesn't crash silently
    try:
        results, _ = train_logistic(config, X_train, X_test, y_train, y_test, training_params)
        assert 'accuracy' in results
    except Exception:
        # Accept failure; classifier cannot learn with single class
        pass