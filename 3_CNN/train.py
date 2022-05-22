import numpy as np
from tqdm import trange
from timeit import default_timer as timer


def train(model, dataset, cost, optimizer, BATCH_SIZE=64, NB_EPOCH=4):

    train_costs, val_costs = [], []

    nb_train_examples = len(dataset.trainx)
    nb_val_examples = len(dataset.valx)

    for epoch in range(NB_EPOCH):

        train_loss = 0
        train_acc = 0

        pbar = trange(nb_train_examples // BATCH_SIZE)
        train_loader = dataset.data_loader("train", BATCH_SIZE)

        start = timer()
        for _, (X_batch, y_batch) in zip(pbar, train_loader):

            y_pred = model.forward(X_batch)
            loss = cost.get(y_pred, y_batch)

            grads = model.backward(y_pred, y_batch)
            params = optimizer.update_params(grads)
            model.set_params(params)

            train_loss += loss * BATCH_SIZE
            train_acc += sum((np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1)))

            pbar.set_description("[Train] Epoch {}".format(epoch + 1))

        end = timer()

        train_loss /= nb_train_examples
        train_costs.append(train_loss)
        train_acc /= nb_train_examples

        info_train = "train-loss: {:0.6f} | train-acc: {:0.3f}"
        print(info_train.format(train_loss, train_acc))
        print(f"Elapsed time for epoch {epoch+1}: {(end-start)/60} min.", end="\n")

        val_loss = 0
        val_acc = 0

        pbar = trange(nb_val_examples // BATCH_SIZE)
        val_loader = dataset.data_loader("val", BATCH_SIZE)

        for i, (X_batch, y_batch) in zip(pbar, val_loader):

            y_pred = model.forward(X_batch)
            loss = cost.get(y_pred, y_batch)

            val_loss += loss * BATCH_SIZE
            val_acc += sum((np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1)))

            pbar.set_description("[Val] Epoch {}".format(epoch + 1))

        val_loss /= nb_val_examples
        val_costs.append(val_loss)
        val_acc /= nb_val_examples

        info_val = "val-loss: {:0.6f} | val-acc: {:0.3f}"
        print(info_val.format(val_loss, val_acc))


    pbar.close()
    return [train_costs, val_costs]
