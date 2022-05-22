from utils import dataloader
from tqdm import trange
import numpy as np


class Metrics:
    def __init__(self, x_test, y_test, y_pred=None):

        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred

    def test(self, model, cost):

        BATCH_SIZE = 64
        self.y_pred_arr = np.array([])
        nb_test_examples = len(self.x_test)
        test_loss = 0
        test_acc = 0

        pbar = trange(nb_test_examples // BATCH_SIZE)
        test_loader = dataloader(self.x_test, self.y_test, BATCH_SIZE)

        for i, (X_batch, y_batch) in zip(pbar, test_loader):

            y_pred = model.forward(X_batch)
            loss = cost.get(y_pred, y_batch)

            test_loss += loss * BATCH_SIZE
            test_acc += sum((np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1)))
            self.y_pred_arr = np.concatenate((self.y_pred_arr, y_pred))
            pbar.set_description("Evaluation")

        test_loss /= nb_test_examples
        test_acc /= nb_test_examples

        info_test = "test-loss: {:0.6f} | test-acc: {:0.3f}"
        print(info_test.format(test_loss, test_acc))

        return test_loss, test_acc, self.y_pred_arr
