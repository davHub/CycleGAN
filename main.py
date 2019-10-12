import time
import random
import os
import argparse
import numpy as np
import tensorflow as tf
import glob
from model import CycleGAN
from tools import *
import imageio
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_data(dir_A, dir_B):
    files_A = glob.glob("{}/*.png".format(dir_A))
    print("#> Dataset A : {} data.".format(len(files_A)))
    files_B = glob.glob("{}/*.png".format(dir_B))
    print("#> Dataset B : {} data.".format(len(files_B)))
    data_A = [preprocess(resize(imageio.imread(f_name))) for f_name in files_A]
    data_B = [preprocess(resize(imageio.imread(f_name))) for f_name in files_B]
    return {"A":data_A, "B":data_B}

def train(sess, model, train_data, saver, from_scratch=True, tot_epochs=200, save_freq=10, test_freq=5, log_freq=20, save_name="model", train_path="./",t_board_path="./"):
    if from_scratch:
        sess.run(tf.global_variables_initializer())
        global_step = 0
    else:
        model.saver.restore(sess, tf.train.latest_checkpoint(save_name))
        #saver.restore(sess, 'model-10')
    len_min = min(len(train_data['A']), len(train_data['B']))
    num_batches = int(len_min/model.batch_size)
    print("#> Number of batches {}".format(num_batches))
    start_time = time.time()
    for epoch in range(tot_epochs):
        random.shuffle(train_data["A"])
        random.shuffle(train_data["B"])
        if epoch > tot_epochs//2:
            decay = model.lr/(tot_epochs//2)
            model.decrease_lr(decay)
            
        for batch in range(6):
            train_A, train_B = [train_data['A'][batch], train_data['B'][batch]]
            if len(train_A.shape)<4:
                train_A = [train_A]
            if len(train_B.shape)<4:
                train_B = [train_B]                
            # Update discriminators A and B
            _, dis_A_loss = sess.run([model.dis_A_optim, model.dis_A_loss], feed_dict={
                model.A_real: train_A,
                model.B_real: train_B
            })

            _, dis_B_loss = sess.run([model.dis_B_optim, model.dis_B_loss], feed_dict={
                model.A_real: train_A,
                model.B_real: train_B
            })

            # We update 2 times generators params
            _, gen_loss = sess.run([model.gen_optim, model.gen_loss], feed_dict={
                model.A_real: train_A,
                model.B_real: train_B
            })

            _, gen_loss = sess.run([model.gen_optim, model.gen_loss], feed_dict={
                model.A_real: train_A,
                model.B_real: train_B
            })
            if (batch+1) % log_freq == 0:
                time_now = time.time()
                t_since_start = (time_now-start_time)/60
                print("#> Ep {} Step {}/{} - ({:3.4}min) -- Losses disA({:.6})  disB({:.6})  gen({:.6})".format(epoch, batch+1, num_batches,t_since_start, dis_A_loss, dis_B_loss, gen_loss))

        if (epoch+1) % test_freq == 0:
            print('#> Testing')
            gen_AB, gen_BA = sess.run([model.B_fake, model.A_fake], feed_dict={
                model.A_real: train_A,
                model.B_real: train_B
            })
            save_img(os.path.join(train_path,"genB_ep{}".format(epoch)), deprocess(gen_AB[0]))
            save_img(os.path.join(train_path,"genA_ep{}".format(epoch)), deprocess(gen_BA[0]))

        if (epoch+1) % save_freq == 0:
            print("#> Saving model")
            saver.save(sess, save_name, global_step=global_step)

        global_step = global_step + 1


def main(args):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print("#> Project dir : {}\n#> Current dir : {}".format(project_dir,args.save_path))
    
    # Create necessary directories
    model_path = os.path.join(args.save_path, "model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_name = os.path.join(model_path, args.model)    
    
    train_path = os.path.join(args.save_path, "train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
        
    t_board_path = os.path.join(args.save_path, "logs")
    if not os.path.exists(t_board_path):
        os.makedirs(t_board_path)
        
    print("#> Building graph model")
    graph = tf.Graph()
    with graph.as_default():
        cycGAN = CycleGAN()
        saver = tf.train.Saver(max_to_keep=6)

    if (not args.testing):
        data = load_data(args.dir_A, args.dir_B)

        with tf.Session(graph=graph) as sess:
            print("#> Training model")
            train(sess, cycGAN, data, saver,
                  save_freq=args.save_freq, test_freq=args.test_freq, log_freq=args.log_freq,
                  save_name=save_name, train_path=train_path,t_board_path=t_board_path)
            
    else:
        print("#> Testing model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # Params for datasets
    parser.add_argument('-dA','--dir_A', dest='dir_A', default='dataset_A', help='Directory name where A images are saved')
    parser.add_argument('-dB','--dir_B', dest='dir_B', default='dataset_B', help='Directory name where B images are saved')
    # Define name save
    parser.add_argument('-m','--model', dest='model', default='CycleGAN', help='Model name')
    parser.add_argument('-s','--save_path', dest='save_path', default=os.getcwd(), help='Directory where to save all')
    # Params for training
    parser.add_argument('-sf','--save_freq', dest='save_freq', type=int, default=10, help="Saving model every 'save_freq' epochs")
    parser.add_argument('-tf','--test_freq', dest='test_freq', type=int, default=5, help="Testing model every 'test_freq' epochs")
    parser.add_argument('-lf','--log_freq', dest='log_freq', type=int, default=20, help="Displaying training log every 'log_freq' steps")
    # If defined, test a model
    parser.add_argument("--testing", help="If defined, do not train but test the model defined", action="store_true")
    args = parser.parse_args()
    main(args)
