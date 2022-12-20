import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Predictor():
    
    def __init__(self):
        pass

    def predictTestSet(self, model, low_res_test, high_res_test, metrics, save_path = ""):
        
        if not len(save_path):
            raise Exception("Provide a path to save the results")

        
        for i in range(len(low_res_test)):
            predicted = np.clip(model.predict(low_res_test[i].reshape(1,256, 256,3)),0.0,1.0).reshape(256, 256,3)
            plt.figure(figsize=(20,10))
            plt.subplot(1,3,1)
            plt.title('High Resolution Img', color = 'green', fontsize = 20)
            plt.imshow(high_res_test[i])
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.title('Low Resolution Img ', color = 'black', fontsize = 20)
            plt.imshow(low_res_test[i])
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.title('Predicted Img ', color = 'red', fontsize = 20)
            plt.imshow(predicted)
            plt.axis('off')
            plt.figtext(0.50, 0.18, "High & Low - SSIM: " + str(round(float(metrics.SSIM(high_res_test[i], low_res_test[i]).numpy() * 100), 3)) + "%", wrap=False, horizontalalignment='center', fontsize=24)
            plt.figtext(0.50, 0.14, "High & Predicted - SSIM: " + str(round(float(metrics.SSIM(high_res_test[i], predicted).numpy() * 100), 3)) + "%", wrap=False, horizontalalignment='center', fontsize=24)
            plt.savefig(save_path + '/CNN_' + str(i) + ".png", facecolor='white', transparent=False)
            plt.close()

    def predictIndividual(self, model, low_res_test, high_res_test, index, metrics):

        if index > len(low_res_test) - 1:
            raise Exception("Provide a valid index")

        predicted = np.clip(model.predict(low_res_test[index].reshape(1,256, 256,3)),0.0,1.0).reshape(256, 256,3)

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.title('High Resolution Img', color = 'green', fontsize = 20)
        plt.imshow(high_res_test[index])
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.title('Low Resolution Img ', color = 'black', fontsize = 20)
        plt.imshow(low_res_test[index])
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title('Predicted Img ', color = 'red', fontsize = 20)
        plt.axis('off')
        plt.figtext(0.50, 0.18, "High & Low - SSIM: " + str(round(float(metrics.SSIM(high_res_test[index], low_res_test[index]).numpy() * 100), 3)) + "%", wrap=False, horizontalalignment='center', fontsize=24)
        plt.figtext(0.50, 0.14, "High & Predicted - SSIM: " + str(round(float(metrics.SSIM(high_res_test[index], predicted).numpy() * 100), 3)) + "%", wrap=False, horizontalalignment='center', fontsize=24)
        plt.imshow(predicted)
        plt.show()
