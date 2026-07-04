"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    # TODO: rescale each column of x to have mean 0 and std 1 (leave zero-std columns alone).
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    sigma[sigma == 0] = 1.0

    x_rescaled = (x - mu) / sigma
    return x_rescaled

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    w = np.zeros(n_features)
    b = 0.0
    return {'w': w, 'b': b}

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    w = params.get("w")
    b = params.get("b")

    score = np.dot(x, w) + b 
    return score

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    return np.where(scores >= 0, 1, -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    return max(0, 1 - y * score)

# Step 6 - svm_objective
def hinge_loss(score, y):
    return np.maximum(0, 1 - y * score)

def svm_objective(x, y, params, reg_lambda):
    score = compute_scores(x, params)
    loss = hinge_loss(score, y)
    w = params.get("w")
    mu = np.mean(loss + reg_lambda * np.dot(w, w), axis=0)
    return mu

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    w = params.get("w")
    score = compute_scores(x, params)
    n = score.shape[0]
    margins = 1 - y * score
    mask = np.where(margins > 0, -1 , 0)
    my = mask * y
    l2 = 2 * reg_lambda * w
    dw = (x.T @ my) * (n ** - 1) + l2
    db = np.sum(my) * (n ** -1)
    return {'dw': dw, 'db': float(db)}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    w = params.get("w")
    b = params.get("b")
    dw = grads.get("dw")
    db = grads.get("db")

    w = w - dw * learning_rate
    b = b - db * learning_rate

    return {"w": w, "b": float(b)}

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    params = initialize_parameters(x.shape[1])

    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
        
    return params

# Step 10 - predict_labels
import numpy as np

def predict_labels(x, params):
    # TODO: return an array of {-1, +1} labels, one per row of x, using params['w'] and params['b'].
    scores = compute_scores(x, params)
    pred = predict_from_scores(scores)
    return pred

# Step 11 - accuracy_score
import numpy as np

def accuracy_score(y_pred, y_true):
    # TODO: return the fraction of positions where y_pred equals y_true.
    arr = np.where(y_pred == y_true, 1, 0)
    mu = np.mean(arr)
    return float(mu)

