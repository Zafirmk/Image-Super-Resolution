import tensorflow as tf

class Metrics:

    def SSIM(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    
    def PSNR(y_true, y_pred):
        max_pixel = 1.0
        return (10.0 * tf.keras.backend.K.log((max_pixel ** 2) / (tf.keras.backend.K.mean(tf.keras.backend.K.square(y_pred - y_true), axis=-1)))) / 2.303