import tensorflow as tf

class Metrics:

    def SSIM(self, y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))