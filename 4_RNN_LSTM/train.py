import numpy as np
import copy

def train_model(
    model,
    loss_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    learning_rate=1e-2,
    epoch_num=1200,
):
    best_val_loss = np.inf
    num_train = len(X_train)
    best_model = None
    for epoch in range(epoch_num):
        shuffled_indices = np.arange(num_train)
        np.random.shuffle(shuffled_indices)
        sections = np.arange(batch_size, num_train, batch_size)
        batches_indices = np.array_split(shuffled_indices, sections)
        batch_losses = np.zeros(len(batches_indices))

        for batch_id, batch_indices in enumerate(batches_indices):
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]

            out = model.forward(batch_X)
            train_loss = loss_fn(out, batch_y)

            grad = out - batch_y

            for param in model.params().values():
                param.grad.fill(0)
            model.backward(grad)

            for param_name, param in model.params().items():
                optimizer = model.optimizers[param_name]
                optimizer.update(param.value, param.grad, learning_rate)

            batch_losses[batch_id] = train_loss

        val_out = model.forward(X_val)
        val_loss = loss_fn(val_out, y_val)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}:  Train loss: {batch_losses.mean():.5f}  Val loss: {val_loss:.5f}"
            )

    return best_model
