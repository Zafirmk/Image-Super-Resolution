import tensorflow as tf

class CNNModel(tf.keras.Model):

    def __init__(self, _shape):
        super(CNNModel, self).__init__()

        self.conv2D_1 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', input_shape = _shape)
        self.conv2D_2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')

        self.dense_1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu')

        self.conv2D_3 = tf.keras.layers.Conv2D(3, 9, padding = 'same', activation = 'relu')

        self.build((None,) + _shape)

    def call(self, x):

        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.conv2D_3(x)

        return x
    
    def summary(self):
        x = tf.keras.layers.Input(shape=(256, 256, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()