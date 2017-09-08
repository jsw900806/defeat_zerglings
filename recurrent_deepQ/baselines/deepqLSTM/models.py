import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers


def _mlp(hiddens, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        q_out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return q_out


def mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)


def _cnn_to_lstm(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=True):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = slim.layers.conv2d(out,
                num_outputs, kernel_size,
                stride=stride,
                activation_fn=tf.nn.relu)

        with tf.variable_scope("action_value"):
            action_out = out
            cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=256, state_is_tuple=True, activation=tf.tanh)
            outputs, _states = tf.nn.dynamic_rnn(cell, action_out, dtype=tf.float32)
            outputs = tf_layers.layer_norm(outputs, scope='layer_norm')
            outputs = tf.contrib.layers.fully_connected(
                outputs[:, -1], num_actions, activation_fn=None)  # We use the last cell's output
            action_scores = layers.fully_connected(outputs, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = out
                cell = tf.contrib.rnn.BasicLSTMCell(
                    num_units=256, state_is_tuple=True, activation=tf.tanh)
                outputs2, _states = tf.nn.dynamic_rnn(cell, state_out, dtype=tf.float32)
                outputs2 = tf_layers.layer_norm(outputs2, scope='layer_norm')
                state_score = tf.contrib.layers.fully_connected(
                    outputs2[:, -1], 1, activation_fn=None)  # We use the last cell's output

            
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def cnn_to_lstm(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_lstm(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)

