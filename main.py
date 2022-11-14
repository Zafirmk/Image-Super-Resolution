from ImageDowngrader import ImageDowngrader
from DataLoader import DataLoader
from Model import CNNModel
import tensorflow as tf

if __name__ == "__main__":
    ImageDowngrader("raw_data/high_res", "raw_data/low_res", 50)
    data_loader = DataLoader("raw_data", [1, 0.9])
    low_res_train, low_res_test, low_res_val, high_res_train, high_res_test, high_res_val = data_loader.loadData()
    model = CNNModel((256,256,3))
    model.summary()
    # model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mse', metrics = ['acc'])
    # model.fit(low_res_train, high_res_train, epochs = 50, batch_size = 1, validation_data = (low_res_val,high_res_val))
    
    print("Shape of training images: ",high_res_train.shape)
    print("Shape of test images: ",high_res_test.shape)
    print("Shape of validation images: ",high_res_val.shape)