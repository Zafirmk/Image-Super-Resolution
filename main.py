from Libraries.ImageDowngrader import ImageDowngrader
from Libraries.DataLoader import DataLoader
from Libraries.Model import CNNModel
from Libraries.Metrics import Metrics
from Libraries.Prediction import Predictor
from Libraries.Graphs import PlotGraphs

import tensorflow as tf

if __name__ == "__main__":
    ImageDowngrader("raw_data/high_res", "raw_data/low_res", 50)
    data_loader = DataLoader("raw_data", [0.9, 0.1])
    metrics = Metrics()
    predictor = Predictor()
    grapher = PlotGraphs()

    low_res_train, low_res_test, low_res_val, high_res_train, high_res_test, high_res_val = data_loader.loadData()
    model = CNNModel((256,256,3))

    choice = input("Do you want to train a new model? (Y/N) - If not, existing weights from model.h5 will be used\n")
    if choice.lower() == 'y':
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mse', metrics = [metrics.SSIM, 'acc'])
        model.fit(low_res_train, high_res_train, epochs = 50, batch_size = 1, validation_data = (low_res_val,high_res_val))
    else:
        model.load_weights("model.h5")

    model.summary()
    
    # Indivdual Prediction being made
    # predictor.predictIndividual(model, low_res_test, high_res_test, metrics, 0)

    # Predict on entire test set
    predictor.predictTestSet(model, low_res_test, high_res_test, metrics, "Images/Predictions")

    print("Shape of training images: ",high_res_train.shape)
    print("Shape of test images: ",high_res_test.shape)
    print("Shape of validation images: ",high_res_val.shape)