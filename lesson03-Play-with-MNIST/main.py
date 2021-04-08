import tensorflow as tf
from tensorflow.keras import layers, optimizers, Sequential, metrics
from pagi.utils.data import tf_get_mnist_dataset

bath_size = 128
ds = tf_get_mnist_dataset(bath_size=bath_size)

network = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(256, activation='relu'),
                      layers.Dense(256, activation='relu'),
                      layers.Dense(10)]
                     )
network.build(input_shape=(None, 28 * 28))
network.summary()

optimizer = optimizers.Adam(learning_rate=0.01)
acc_meter = metrics.Accuracy()

for iter in range(20):
    for step, (x, y) in enumerate(ds):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28 * 28))
            out = network(x)
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.square(out - y_onehot)
            loss = tf.reduce_sum(loss) / bath_size
        acc_meter.update_state(tf.argmax(out, axis=1), y)
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
    #
    print(iter, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
    acc_meter.reset_states()
