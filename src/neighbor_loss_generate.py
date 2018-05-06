"""
Figure something out
"""

import click
import numpy as np
import tensorflow as tf
from benchmarks import benchmark_func
from utils.data_utils import stream_json, BASE_PATH

tf.enable_eager_execution()


@click.group()
def cli():
    pass


def load_X_Y(dataset, split, encoding_name):
    json_file = BASE_PATH / dataset / encoding_name /\
        ('%s-encoded-captions-and-images.json' % split)

    X = [np.array(i['x_image'], dtype=np.float32) for i in stream_json(json_file)]
    X = np.array(X)
    Y = [np.array(i['x_text'], dtype=np.float32) for i in stream_json(json_file)]
    Y = np.array(Y)
    return X, Y


def save(X_c, Y_c, split, model, dataset, encoding_name, model_name='test1'):
    """Save the encoded components and the model."""
    path = BASE_PATH / dataset / model_name / encoding_name
  
    path.mkdir(exist_ok=True, parents=True)

    np.save(path / ('{}_X_c'.format(split)), X_c)
    np.save(path / ('{}_Y_c'.format(split)), Y_c)


def minibatch_generator(X, Y, batch_size=32, epochs=1000):
    
    for _ in range(int(epochs)):
        idxs = np.random.choice(X.shape[0], size=batch_size * 2, replace=False)
        pos_idxs = idxs[:batch_size]
        neg_idxs = idxs[batch_size:]
        x = X[pos_idxs]
        y = Y[pos_idxs]

        # some negative samples for X might be identical to each other
        # since we have duplicates (there are 5 captions per image)
        # but this shouldn't affect the loss since MSE will be 0
        x_neg = X[neg_idxs]
        y_neg = Y[neg_idxs]

        yield x, y, x_neg, y_neg


@click.command()
@click.argument('dataset')
@click.argument('encoding_name')
def train(dataset, encoding_name):

    X, Y = load_X_Y(dataset, 'train', encoding_name)
    X_val, Y_val = load_X_Y(dataset, 'validation', encoding_name)

    # we should be able to devise a linear method that performs as well as cca

    embedding_dim = 300
    lambda_c = 1
    lambda_x = 1e-3  # weight on X distances for negative samples
    lambda_y = 1  # weight on Y distances for negative samples

    U = tf.get_variable(
        "U", [embedding_dim, 4096], dtype=tf.float32,
        initializer=tf.keras.initializers.glorot_uniform())
    V = tf.get_variable(
        "V", [embedding_dim, 300], dtype=tf.float32,
        initializer=tf.keras.initializers.glorot_uniform())

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    def get_loss(x, y, x_neg, y_neg, U, V):
        Y_norm = y / tf.reshape(tf.norm(y, axis=1), (-1, 1))
        Y_prime_norm = y_neg / tf.reshape(tf.norm(y_neg, axis=1), (-1, 1))

        Ux = tf.matmul(U, x.T)
        Vy = tf.matmul(V, y.T)

        Ux_norm = Ux / tf.norm(Ux, axis=0)
        Vy_norm = Vy / tf.norm(Vy, axis=0)

        cosine_dist_loss1 = tf.reduce_sum(tf.multiply(Ux_norm, Vy_norm), axis=0)
        cosine_dist_loss = -lambda_c * tf.reduce_mean(cosine_dist_loss1)

        # # loss for negative samples
        Ux_prime = tf.matmul(U, x_neg.T)
        Vy_prime = tf.matmul(V, y_neg.T)
        Vy_prime_norm = Vy_prime / tf.norm(Vy_prime, axis=0)

        y_neg_dist = tf.reduce_sum(tf.multiply(Y_prime_norm, Y_norm), axis=1)
        vy_neg_dist = tf.reduce_sum(tf.multiply(Vy_prime_norm, Vy_norm), axis=0)
        neg_y_loss = lambda_y * tf.reduce_sum(tf.square(y_neg_dist - vy_neg_dist))

        ux_neg_dist = tf.reduce_mean(tf.square(Ux_prime - Ux), axis=0)
        x_neg_dist = tf.reduce_mean(tf.square(x_neg - x), axis=1)
        neg_x_loss = lambda_x * tf.losses.mean_squared_error(x_neg_dist, ux_neg_dist)

        loss = cosine_dist_loss + neg_y_loss + neg_x_loss
        return loss, cosine_dist_loss, neg_y_loss, neg_x_loss

    for idx, (x, y, x_neg, y_neg) in enumerate(minibatch_generator(X, Y, epochs=1e5)):

        with tf.contrib.eager.GradientTape() as tape:
            loss, *rest = get_loss(x, y, x_neg, y_neg, U, V)

        grads = tape.gradient(loss, [U, V])

        if not idx % 200:
            print(idx, loss, list(map(lambda x: x.numpy(), rest)))

            loss, *rest = get_loss(X_val, Y_val, X_val, Y_val, U, V)
            print('val cosine distance', rest[0].numpy())
            X_val_c, Y_val_c = U.numpy().dot(X_val.T).T, V.numpy().dot(Y_val.T).T
            benchmark_func(dataset, encoding_name, 'validation', X_val_c, Y_val_c, 'cosine')

        optimizer.apply_gradients(
            zip(grads, [U, V]),
            global_step=tf.train.get_or_create_global_step())


cli.add_command(train)


if __name__ == '__main__':
    cli()
