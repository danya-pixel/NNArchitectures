import numpy as np
import matplotlib.pyplot as plt
import imageio

def sample_images(batch_size, image_size, image_dir, filenames, images, epoch, show):
        images = np.reshape(images, (batch_size, image_size, image_size))

        fig = plt.figure(figsize=(4, 4))

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")

        current_epoch_filename = image_dir.joinpath(f"GAN_epoch{epoch}.png")
        filenames.append(current_epoch_filename)
        plt.savefig(current_epoch_filename)

        if show == True:
            plt.show()
        else:
            plt.close()


def show_batch(train_X, train_y):
    num_row = 2
    num_col = 5
    _, axes = plt.subplots(num_row, num_col, figsize=(2 * num_col, 2 * num_row))
    for i in range(10):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(train_X[i], cmap="gray")
        ax.set_title("Label: {}".format(train_y[i]))
    plt.tight_layout()
    plt.show()