import tensorflow as tf

def get_loss():
    return tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.05
    )

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)

        loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)

        return tf.reduce_mean(loss)

    return loss_fn