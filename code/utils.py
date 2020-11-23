import pickle
import numpy as np
import tensorflow as tf
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


def predict_on_batch(sess, predict_func, test_x, test_t, test_g, batchsize=800):
    n_samples_test = test_x.shape[0]
    test_pred = np.zeros(n_samples_test)
    batch = [(start, min(start + batchsize, n_samples_test)) for start in range(0, n_samples_test, batchsize)]
    for (start, end) in batch:
        batch_x = test_x.iloc[start:end]
        batch_t = test_t[start:end]
        batch_g = test_g[start:end]
        _pred = predict_func(sess, batch_x, batch_t, batch_g)
        test_pred[start:end] = _pred.reshape(-1)
    return test_pred
