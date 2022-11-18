import keras
import numpy as np
import tensorflow as tf


def non_neg_mse(y: tf.Tensor, y_hat: tf.Tensor):
    mse = (y - y_hat) ** 2
    positive = 10 * tf.maximum(-y_hat, 0.0)
    return tf.reduce_mean(mse + positive)


def predict_next_note(
        notes: np.ndarray, 
        model: keras.Model, 
        temperature: float = 1.0) -> int:
    """Generates a note IDs using a trained sequence model."""

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    start = predictions['start']
    end = predictions['end']
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    start = tf.squeeze(start, axis=-1)
    end = tf.squeeze(end, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    start = tf.maximum(0, start)
    end = tf.maximum(0, end)
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return (int(pitch), float(start), float(end), float(step), float(duration))
