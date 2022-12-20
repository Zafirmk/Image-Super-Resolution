import matplotlib.pyplot as plt
import numpy as np

class PlotGraphs():

    def __init__(self) -> None:
        pass

    def plot_TrainingValidation_acc(self, model, save_path):
        plt.figure(figsize=(10,5))
        plt.plot(model.history.history['acc'], label = "Training Accuracy")
        plt.plot(model.history.history['val_acc'], label = "Validation Accuracy")
        plt.legend(loc="lower right")
        plt.xlabel("Epochs")
        plt.title("Training Accuracy | Validation Accuracy")
        plt.savefig(save_path + "/CNN_Accuracy.png", facecolor='white', transparent=False)
    
    def plot_TrainingValidation_loss(self, model, save_path):
        plt.figure(figsize=(10,5))
        plt.plot(model.history.history['loss'], label = "Training Loss")
        plt.plot(model.history.history['val_loss'], label = "Validation Loss")
        plt.legend(loc="upper right")
        plt.xlabel("Epochs")
        plt.title("Training Loss | Validation Loss")
        plt.savefig(save_path + "/CNN_Loss.png", facecolor='white', transparent=False)

    def plot_TrainingValidation_SSIM(self, model, save_path):
        plt.figure(figsize=(10,5))
        plt.plot(model.history.history['SSIM'], label = "Training SSIM")
        plt.plot(model.history.history['val_SSIM'], label = "Validation SSIM")
        plt.legend(loc = "lower right")
        plt.xlabel("Epochs")
        plt.title("Training SSIM | Validation SSIM")
        plt.savefig(save_path + "/CNN_Training_SSIM.png", facecolor='white', transparent=False)

    def plot_TestSet_SSIM(self, model, low_res_test, high_res_test, metrics, save_path):

        pred_high_scores = []
        low_high_scores = []

        for i in range(len(low_res_test)):
            predicted = np.clip(model.predict(low_res_test[i].reshape(1,256, 256,3)),0.0,1.0).reshape(256, 256,3)
            pred_high_scores.append(metrics.SSIM(high_res_test[i], predicted).numpy())
            low_high_scores.append(metrics.SSIM(high_res_test[i], low_res_test[i]).numpy())

        limit = 20 if len(pred_high_scores) > 20 else len(pred_high_scores)

        plt.figure(figsize=(10,5))
        plt.title("Test Set SSIM Results")
        plt.plot(pred_high_scores[:limit], label = "Prediction & High Quality Image")
        plt.plot(low_high_scores[:limit], label = "Low & High Quality Image")
        plt.xlabel("Test Image Number")
        plt.xticks([i+1 for i in range(len(low_res_test[:limit]))])
        plt.legend()
        plt.savefig(save_path + "/CNN_Test_SSIM.png", facecolor='white', transparent=False)
