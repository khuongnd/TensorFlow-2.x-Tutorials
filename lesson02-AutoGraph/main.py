import tensorflow as tf
import timeit

cell = tf.keras.layers.LSTMCell(10)


@tf.function
def fn(input, state):
    '''
    Calculate output of cell (input, state)
    '''
    return cell(input, state)


input = tf.zeros([10, 10])
state = [tf.ones([10, 10])] * 2

# warmup
cell(input, state), fn(input, state)

dynamic_graph_time = timeit.timeit(lambda: cell(input, state), number=100)
static_graph_time = timeit.timeit(lambda: fn(input, state), number=100)
print('dynamic_graph_time:', dynamic_graph_time)
print('static_graph_time:', static_graph_time)
