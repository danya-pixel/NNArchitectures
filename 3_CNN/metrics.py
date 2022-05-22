from tqdm import trange
from utils import dataloader
from layers import Sigmoid
import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self, x_test, y_test, y_pred=None, y_pred_arr_probs=None):

        self.x_test = x_test
        self.y_test = y_test
        self.y_pred_arr = y_pred
        self.y_pred_arr_probs = y_pred_arr_probs

    def test(self, model, cost):

        BATCH_SIZE = 16
        self.y_pred_arr = np.array([])
        self.y_pred_arr_probs = np.zeros((0, 10))
        nb_test_examples = len(self.x_test)
        test_loss = 0
        test_acc = 0

        pbar = trange(nb_test_examples // BATCH_SIZE)
        test_loader = dataloader(self.x_test, self.y_test, BATCH_SIZE)

        for _, (X_batch, y_batch) in zip(pbar, test_loader):

            y_pred_probs = model.forward(X_batch)
            loss = cost.get(y_pred_probs, y_batch)
            test_loss += loss * BATCH_SIZE
            test_acc += sum(
                (np.argmax(y_batch, axis=1) == np.argmax(y_pred_probs, axis=1))
            )

            y_pred = np.argmax(y_pred_probs, axis=1)
            self.y_pred_arr_probs = np.concatenate(
                (self.y_pred_arr_probs, y_pred_probs), axis=0
            )
            self.y_pred_arr = np.concatenate((self.y_pred_arr, y_pred))

            pbar.set_description("Evaluation")

        test_loss /= nb_test_examples
        test_acc /= nb_test_examples

        info_test = "test-loss: {:0.6f} | test-acc: {:0.3f}"
        print(info_test.format(test_loss, test_acc))

        return test_loss, test_acc, self.y_pred_arr


    def ROC_AUC(self):
        points_num = 300
        thresholds = np.linspace(0.5, 1, points_num)
        sigmoid_layer = Sigmoid()
        y_sigm = sigmoid_layer.forward(self.y_pred_arr_probs)
        y_pred_bin_l = (y_sigm == y_sigm.max(axis=1)[:, None]).astype(int)
        plt.figure(figsize=(20, 20))
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
        )

        for label in range(10):
            rec = []
            fpr = []
            for _, threshold in enumerate(thresholds):
                y_test = self.y_test[:, label]
                y_pred_bin = y_pred_bin_l[:, label] * y_sigm[:, label]

                TP = np.sum(np.logical_and(y_pred_bin > threshold, y_test == 1))

                TN = np.sum(np.logical_and(y_pred_bin < threshold, y_test == 0))

                FP = np.sum(np.logical_and(y_pred_bin > threshold, y_test == 0))

                recall = TP / np.sum(np.argmax(self.y_test, axis=1) == label)
                FPR = FP / (FP + TN)
                rec.append(recall)
                fpr.append(FPR)

            plt.subplot(10, 2, label + 1)
            plt.plot(fpr, rec)
            plt.title(f"class: {label}")