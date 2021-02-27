import os
from pagi.utils.data import tf_get_mnist_dataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )


@tf.function
def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


def train(epoch, model, optimizer):
    train_ds = tf_get_mnist_dataset()
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = compute_loss(logits, y)
        # compute gradient
        grads = tape.gradient(loss, model.trainable_variables)
        # update to weights
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        accuracy = compute_accuracy(logits, y)
        # print log
        if step % 500 == 0:
            print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

    return loss, accuracy


if __name__ == "__main__":
    # Init
    model = keras.Sequential([
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(100),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(100),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(10)
    ])
    optimizer = optimizers.Adam()
    # Train model
    for epoch in range(20):
        loss, accuracy = train(epoch, model, optimizer)
    # print log
    print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
