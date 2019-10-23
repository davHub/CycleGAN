import time
import random
import os
import argparse
import numpy as np
import tensorflow as tf
from model import CycleGAN
from tools import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(args):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print("#> Project dir : {}\n#> Current dir : {}".format(
        project_dir, args.save_path))

    # Create necessary directories
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Build the graph
    print("#> Building graph model")
    graph = tf.Graph()
    with graph.as_default():
        cycGAN = CycleGAN(img_shape=[
                          args.fine_size, args.fine_size, 3], color_reg=args.color_reg, testing=args.testing,norm=args.norm)
        saver = tf.train.Saver(max_to_keep=6)


    if (not args.testing):
        # Create pathfor training
        model_path = os.path.join(args.save_path, "model")
        train_path = os.path.join(args.save_path, "train")
        t_board_path = os.path.join(args.save_path, "logs")
        save_name = os.path.join(model_path, args.model)
        restore_name = os.path.join(
            model_path, args.restore) if args.restore is not None else None

        # Create necessary directories
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(t_board_path):
            os.makedirs(t_board_path)

        # Load data
        data = load_data(args.dir_A, args.dir_B, resize_dim=args.load_size)

        # Start training
        with tf.Session(graph=graph) as sess:
            print("#> Training model")
            cycGAN.train(sess, data, saver, tot_epochs=args.nb_epochs,
                         save_freq=args.save_freq, test_freq=args.test_freq, log_freq=args.log_freq,
                         save_name=save_name, train_path=train_path, t_board_path=t_board_path, model_path=restore_name)

    else:
        # Create path testing
        model_path = os.path.join(args.save_path, "model")
        test_path = os.path.join(args.save_path, "test")
        restore_name = os.path.join(model_path, args.restore)

        # Create necessary directories
        if not os.path.exists("{}".format(model_path)):
            raise FileNotFoundError("Model {} : not found".format(save_name))
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        # Load data
        data = load_data(args.dir_A, args.dir_B, resize_dim=args.fine_size)

        # Start testing
        with tf.Session(graph=graph) as sess:
            print("#> Testing model")
            cycGAN.test(sess, data, saver, num_test=15,
                        test_path=test_path, model_path=restore_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # Params for datasets
    parser.add_argument('-dA', '--dir_A', dest='dir_A', default='dataset_A', help='Directory name where A images are saved')
    parser.add_argument('-dB', '--dir_B', dest='dir_B', default='dataset_B', help='Directory name where B images are saved')
    # Define name save
    parser.add_argument('-m', '--model', dest='model', default='CycleGAN', help='Model name')
    parser.add_argument('-s', '--save_path', dest='save_path', default=os.getcwd(), help='Directory where to save all')
    parser.add_argument('--restore', dest='restore', default=None, help='Model to restore')
    # Params to change training parameters
    parser.add_argument('-e', '--nb_epochs', dest='nb_epochs', type=int, default=200, help='Nb epochs for training')
    parser.add_argument('-n', '--norm', dest='norm', default='instance', help='Normalization to use')
    parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help="Dimension of the data")
    parser.add_argument('--load_size', dest='load_size', type=int, default=286, help="Dimension of the data before cropping")
    parser.add_argument("--color_reg", help="If defined, add the color regularization.", action="store_true")
    # Params for training config
    parser.add_argument('-sf', '--save_freq', dest='save_freq', type=int, default=10, help="Saving model every 'save_freq' epochs")
    parser.add_argument('-tf', '--test_freq', dest='test_freq', type=int, default=5, help="Testing model every 'test_freq' epochs")
    parser.add_argument('-lf', '--log_freq', dest='log_freq', type=int, default=100, help="Displaying training log every 'log_freq' steps")
    # If defined, test a model
    parser.add_argument("--testing", help="If defined, do not train but test the model defined.", action="store_true")
    args = parser.parse_args()
    main(args)
