import numpy as np
from pathlib import Path
from activations import sigmoid, dsigmoid, lrelu, dlrelu, dtanh
from vizualize import sample_images, generate_gif


class GAN:
    def __init__(
        self,
        numbers,
        epochs=100,
        batch_size=64,
        input_layer_size_g=100,
        hidden_layer_size_g=128,
        hidden_layer_size_d=128,
        learning_rate=1e-3,
        decay_rate=1e-4,
        image_size=28,
        display_epochs=5,
        create_gif=True,
    ):
        self.numbers = numbers
        self.epochs = epochs
        self.batch_size = batch_size
        self.nx_g = input_layer_size_g
        self.nh_g = hidden_layer_size_g
        self.nh_d = hidden_layer_size_d
        self.lr = learning_rate
        self.dr = decay_rate
        self.image_size = image_size
        self.display_epochs = display_epochs
        self.create_gif = create_gif

        self.image_dir = Path("./GAN_sample_images")
        if not self.image_dir.is_dir():
            self.image_dir.mkdir()

        self.filenames = []

        self.W0_g = np.random.randn(self.nx_g, self.nh_g) * np.sqrt(2.0 / self.nx_g)
        self.b0_g = np.zeros((1, self.nh_g))

        self.W1_g = np.random.randn(self.nh_g, self.image_size**2) * np.sqrt(
            2.0 / self.nh_g
        )
        self.b1_g = np.zeros((1, self.image_size**2))

        self.W0_d = np.random.randn(self.image_size**2, self.nh_d) * np.sqrt(
            2.0 / self.image_size**2
        )
        self.b0_d = np.zeros((1, self.nh_d))

        self.W1_d = np.random.randn(self.nh_d, 1) * np.sqrt(2.0 / self.nh_d)
        self.b1_d = np.zeros((1, 1))

    def forward_generator(self, z):
        """
        Generator forward pass
        """
        self.z0_g = np.dot(z, self.W0_g) + self.b0_g
        self.a0_g = lrelu(self.z0_g, alpha=0)

        self.z1_g = np.dot(self.a0_g, self.W1_g) + self.b1_g
        self.a1_g = np.tanh(self.z1_g)
        return self.z1_g, self.a1_g

    def forward_discriminator(self, x):
        """
        Discriminator forward pass
        """
        self.z0_d = np.dot(x, self.W0_d) + self.b0_d
        self.a0_d = lrelu(self.z0_d)

        self.z1_d = np.dot(self.a0_d, self.W1_d) + self.b1_d
        self.a1_d = sigmoid(self.z1_d)
        return self.z1_d, self.a1_d

    def backward_discriminator(
        self, x_real, z1_real, a1_real, x_fake, z1_fake, a1_fake
    ):
        """
        Backprop for discriminator
        """

        da1_real = -1.0 / (a1_real + 1e-8)

        dz1_real = da1_real * dsigmoid(z1_real)
        dW1_real = np.dot(self.a0_d.T, dz1_real)
        db1_real = np.sum(dz1_real, axis=0, keepdims=True)

        da0_real = np.dot(dz1_real, self.W1_d.T)
        dz0_real = da0_real * dlrelu(self.z0_d)
        dW0_real = np.dot(x_real.T, dz0_real)
        db0_real = np.sum(dz0_real, axis=0, keepdims=True)

        da1_fake = 1.0 / (1.0 - a1_fake + 1e-8)

        dz1_fake = da1_fake * dsigmoid(z1_fake)
        dW1_fake = np.dot(self.a0_d.T, dz1_fake)
        db1_fake = np.sum(dz1_fake, axis=0, keepdims=True)

        da0_fake = np.dot(dz1_fake, self.W1_d.T)
        dz0_fake = da0_fake * dlrelu(self.z0_d, alpha=0)
        dW0_fake = np.dot(x_fake.T, dz0_fake)
        db0_fake = np.sum(dz0_fake, axis=0, keepdims=True)

        dW1 = dW1_real + dW1_fake
        db1 = db1_real + db1_fake

        dW0 = dW0_real + dW0_fake
        db0 = db0_real + db0_fake

        self.W0_d -= self.lr * dW0
        self.b0_d -= self.lr * db0

        self.W1_d -= self.lr * dW1
        self.b1_d -= self.lr * db1

    def backward_generator(self, z, x_fake, z1_fake, a1_fake):
        """
        Generator backprop
        """

        da1_d = -1.0 / (a1_fake + 1e-8)

        dz1_d = da1_d * dsigmoid(z1_fake)
        da0_d = np.dot(dz1_d, self.W1_d.T)
        dz0_d = da0_d * dlrelu(self.z0_d)
        dx_d = np.dot(dz0_d, self.W0_d.T)

        dz1_g = dx_d * dtanh(self.z1_g)
        dW1_g = np.dot(self.a0_g.T, dz1_g)
        db1_g = np.sum(dz1_g, axis=0, keepdims=True)

        da0_g = np.dot(dz1_g, self.W1_g.T)
        dz0_g = da0_g * dlrelu(self.z0_g, alpha=0)
        dW0_g = np.dot(z.T, dz0_g)
        db0_g = np.sum(dz0_g, axis=0, keepdims=True)

        self.W0_g -= self.lr * dW0_g
        self.b0_g -= self.lr * db0_g

        self.W1_g -= self.lr * dW1_g
        self.b1_g -= self.lr * db1_g

    def preprocess_data(self, x, y):
        x_train = []
        y_train = []

        for i in range(y.shape[0]):
            if y[i] in self.numbers:
                x_train.append(x[i])
                y_train.append(y[i])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        num_batches = x_train.shape[0] // self.batch_size
        x_train = x_train[: num_batches * self.batch_size]
        y_train = y_train[: num_batches * self.batch_size]

        x_train = np.reshape(x_train, (x_train.shape[0], -1))

        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        idx = np.random.permutation(len(x_train))
        x_train, y_train = x_train[idx], y_train[idx]
        return x_train, y_train, num_batches

    def train(self, x, y):
        J_Ds = []
        J_Gs = []
        x_train, _, num_batches = self.preprocess_data(x, y)

        for epoch in range(self.epochs):
            for i in range(num_batches):
                x_real = x_train[i * self.batch_size : (i + 1) * self.batch_size]
                z = np.random.normal(0, 1, size=[self.batch_size, self.nx_g])

                z1_g, x_fake = self.forward_generator(z)

                z1_d_real, a1_d_real = self.forward_discriminator(x_real)
                z1_d_fake, a1_d_fake = self.forward_discriminator(x_fake)

                J_D = np.mean(-np.log(a1_d_real) - np.log(1 - a1_d_fake))
                J_Ds.append(J_D)

                J_G = np.mean(-np.log(a1_d_fake))
                J_Gs.append(J_G)

                self.backward_discriminator(
                    x_real, z1_d_real, a1_d_real, x_fake, z1_d_fake, a1_d_fake
                )
                self.backward_generator(z, x_fake, z1_d_fake, a1_d_fake)

            if epoch % self.display_epochs == 0:
                print(
                    f"Epoch:{epoch:} G loss:{J_G:.4f} D loss:{J_D:.4f}"
                )
                sample_images(
                    self.batch_size,
                    self.image_size,
                    self.image_dir,
                    self.filenames,
                    x_fake,
                    epoch,
                    show=True,
                )
            else:
                sample_images(
                    self.batch_size,
                    self.image_size,
                    self.image_dir,
                    self.filenames,
                    x_fake,
                    epoch,
                    show=False,
                )

            self.lr = self.lr * (1.0 / (1.0 + self.dr * epoch))

        if self.create_gif:
            generate_gif(self.filenames)
        return J_Ds, J_Gs
