import time
import random
import os
import argparse
import numpy as np
import tensorflow as tf
from model import CycleGAN
from tools import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

''' 

def train(sess, model, train_data, saver, tot_epochs=200, save_freq=10, test_freq=5, log_freq=20, save_name="model", train_path="./", t_board_path="./", model_path=None, crop_size=256):
    """ Train a model

    Args:
        sess: tensorflow session
        model: ML model to train
        train_data: dataset
        saver: tf.Saver instance
        tot_epochs: number of epochs to train the model
        save_freq: the frequency of saving model
        test_freq: the frequency of testing the model
        log_freq: the frequency of logging training info
        save_name: the name of the model to be saved
        train_path: the path to the train directory (used to print data)
        t_board_path: not used        
        model_path: the path to a pretrained model
    """
    # Initialise model weights thanks to either random or pretrained values
    global_step = 0
    if model_path is None:
        sess.run(tf.global_variables_initializer())
    else:
        # saver.restore(sess, tf.train.latest_checkpoint(save_name))
        saver.restore(sess, model_path)

    # Calculate number of batches by epoch (size of batch 1 --> size min of the datasets)
    num_batches = min(len(train_data['A']), len(train_data['B']))
    if num_batches == 0:
        raise ValueError()
    print("#> Number of batches {}".format(num_batches))

    # Get image to test and analyse training
    n_test = 8
    random.shuffle(train_data['A'])
    random.shuffle(train_data['B'])
    test_A = train_data['A'][:n_test]
    test_B = train_data['B'][:n_test]

    start_time = time.time()
    for epoch in range(tot_epochs):
        # Shuffle datasets
        random.shuffle(train_data["A"])
        random.shuffle(train_data["B"])

        # Decay learning rate similarly than in the paper
        if epoch > tot_epochs//2:
            decay = model.lr/(tot_epochs//2)
            model.decrease_lr(decay)

        for step in range(num_batches):
            # Get next batch
            train_A = make_noisy(train_data['A'][step], crop_size=crop_size)
            train_B = make_noisy(train_data['B'][step], crop_size=crop_size)

            if len(train_A.shape) < 4:
                train_A = [train_A]
            if len(train_B.shape) < 4:
                train_B = [train_B]

            # We update generators params
            _, gen_loss, A_fake, B_fake = sess.run([model.gen_optim, model.gen_loss, model.A_fake, model.B_fake], feed_dict={
                                                   model.A_real: train_A, model.B_real: train_B, model.lr_pl: model.lr})

            # Update discriminators params
            _, dis_loss = sess.run([model.dis_optim, model.dis_loss], feed_dict={model.A_real: train_A, model.B_real: train_B,
                                                                                 model.A_fake_buff: model.buffer_fake_A(A_fake), model.B_fake_buff: model.buffer_fake_B(B_fake),
                                                                                 model.lr_pl: model.lr})

            # Display log every 'log_freq' steps
            if (step+1) % log_freq == 0:
                time_now = time.time()
                t_since_start = (time_now-start_time)/60
                print("#> Ep {} Step {}/{} - ({:3.4}min) -- Losses dis({:.6})  gen({:.6})".format(
                    epoch, step+1, num_batches, t_since_start, dis_loss, gen_loss))

        # Test model every 'test_freq' epochs
        if (epoch+1) % test_freq == 0:
            print('#> Testing')
            gen_AB, gen_BA = sess.run([model.B_fake, model.A_fake], feed_dict={
                                      model.A_real: test_A, model.B_real: test_B})
            save_img(os.path.join(train_path, "genAB_ep{}".format(
                epoch)), group_images(deprocess(gen_AB)))
            save_img(os.path.join(train_path, "genBA_ep{}".format(
                epoch)), group_images(deprocess(gen_BA)))

        # Save model every 'save_freq' epochs
        if (epoch+1) % save_freq == 0:
            print("#> Saving model")
            saver.save(sess, save_name, global_step=global_step)

        global_step = global_step + 1


def test(sess, model, test_data, saver, num_test=30, test_path="./", model_path=None):
    """ Test a model

    Args:
        sess: tensorflow session
        model: ML model to train
        test_data: the dataset
        saver: the saver
        num_test: number of data from test dataset to use for testing
        test_path: the path to the test directory (used to print test results)
        model_path: the path to the trained model
    """
    saver.restore(sess, model_path)

    random.shuffle(test_data["A"])
    random.shuffle(test_data["B"])
    num_test = min(min(len(test_data["A"]), len(test_data["B"])), num_test)
    test_A, test_B = [test_data['A'][:num_test], test_data['B'][:num_test]]
    if len(np.array(test_A).shape) < 4:
        test_A = [test_A]
    if len(np.array(test_B).shape) < 4:
        test_B = [test_B]

    A_fake, B_fake, A_cyc, B_cyc = sess.run([model.A_fake, model.B_fake, model.A_cyc, model.B_cyc], feed_dict={
                                            model.A_real: test_A, model.B_real: test_B})

    for i in range(num_test):
        im_to_save_1 = np.concatenate(
            (deprocess(test_A[i]), deprocess(B_fake[i]), deprocess(A_cyc[i])), axis=1)
        save_img(os.path.join(test_path, "test_AB_{}".format(i)), im_to_save_1)
        im_to_save_2 = np.concatenate(
            (deprocess(test_B[i]), deprocess(A_fake[i]), deprocess(B_cyc[i])), axis=1)
        save_img(os.path.join(test_path, "test_BA_{}".format(i)), im_to_save_2)

'''

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
                          args.fine_size, args.fine_size, 3], color_reg=args.color_reg, testing=args.testing)
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
