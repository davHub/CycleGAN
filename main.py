import time
import random
import os
import argparse
import numpy as np
import tensorflow as tf
import glob
from model import CycleGAN
from tools import *
import cv2
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_data(dir_A, dir_B, file_type='png'):
    """ Load data images from 2 directories
    
    Args:
        dir_A: directory of the dataset A
        dir_B: directory of the dataset B
        file_type: type of the image files ('png', 'jpg' or 'gif')
    """
    def gray_scale(x):
        return  len(x.shape)<3 or (len(x.shape)==3 and x.shape[-1]<3)
    
    # Get data filename for both dataset A and B
    files_A = glob.glob("{}/*.{}".format(dir_A, file_type))
    print("#> Dataset A from {}: {} data.".format(dir_A, len(files_A)))
    files_B = glob.glob("{}/*.{}".format(dir_B, file_type))
    print("#> Dataset B from {}: {} data.".format(dir_B, len(files_B)))

    # Preprocess images loaded
    data_A = [preprocess(resize(cv2.imread(f_name, cv2.IMREAD_COLOR))) for f_name in files_A]
    data_B = [preprocess(resize(cv2.imread(f_name, cv2.IMREAD_COLOR))) for f_name in files_B]

    # keep only non-grayscale images
    data_A = [x for x in data_A if not gray_scale(x)]    
    data_B = [x for x in data_B if not gray_scale(x)]
    return {"A":data_A, "B":data_B}

def train(sess, model, train_data, saver, tot_epochs=200, save_freq=10, test_freq=5, log_freq=20, save_name="model", train_path="./",t_board_path="./", model_path=None):
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
    if model_path is None:
        sess.run(tf.global_variables_initializer())
        global_step = 0
    else:
        # saver.restore(sess, tf.train.latest_checkpoint(save_name))
        saver.restore(sess, model_path)

    # Calculate number of batches by epoch (size of batch 1 --> size min of the datasets)
    num_batches = min(len(train_data['A']), len(train_data['B']))
    print("#> Number of batches {}".format(num_batches))
    
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
            train_A, train_B = [train_data['A'][step], train_data['B'][step]]
            
            if len(train_A.shape)<4: train_A = [train_A]
            if len(train_B.shape)<4: train_B = [train_B]
                            
            # We update generators params
            _, gen_loss, A_fake, B_fake = sess.run([model.gen_optim, model.gen_loss, model.A_fake, model.B_fake], feed_dict={ model.A_real: train_A, model.B_real: train_B, model.lr_pl: model.lr })
            model.buffer_fake_A.append(A_fake)
            model.buffer_fake_B.append(B_fake)
           
            # Update discriminators params
            _, dis_loss = sess.run([model.dis_optim, model.dis_loss], feed_dict={   model.A_real: train_A, model.B_real: train_B,
                                                                                    model.A_fake_buff: train_A, model.B_fake_buff: train_B,
                                                                                    model.lr_pl: model.lr})

            # Display log every 'log_freq' steps  
            if (step+1) % log_freq == 0:
                time_now = time.time()
                t_since_start = (time_now-start_time)/60
                print("#> Ep {} Step {}/{} - ({:3.4}min) -- Losses dis({:.6})  gen({:.6})".format(epoch, step+1, num_batches,t_since_start, dis_loss, gen_loss))
        
        # Test model every 'test_freq' epochs  
        if (epoch+1) % test_freq == 0:
            print('#> Testing')
            gen_AB, gen_BA = sess.run([model.B_fake, model.A_fake], feed_dict={ model.A_real: train_A, model.B_real: train_B })
            save_img(os.path.join(train_path,"genAB_ep{}".format(epoch)), deprocess(gen_AB[0]))
            save_img(os.path.join(train_path,"genBA_ep{}".format(epoch)), deprocess(gen_BA[0]))

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
    num_test = min(min(len(test_data["A"]), len(test_data["B"])),num_test)
    test_A, test_B = [test_data['A'][:num_test], test_data['B'][:num_test]]
    if len(np.array(test_A).shape)<4: test_A = [test_A]
    if len(np.array(test_B).shape)<4: test_B = [test_B]
    
    A_fake, B_fake, A_cyc, B_cyc = sess.run([model.A_fake, model.B_fake, model.A_cyc, model.B_cyc], feed_dict={ model.A_real: test_A, model.B_real: test_B })
    
    for i in range(num_test):
        im_to_save_1 = np.concatenate((deprocess(test_A[i]), deprocess(B_fake[i]), deprocess(A_cyc[i])), axis=1)
        save_img(os.path.join(test_path,"test_AB_{}".format(i)), im_to_save_1)
        im_to_save_2 = np.concatenate((deprocess(test_B[i]), deprocess(A_fake[i]), deprocess(B_cyc[i])), axis=1)
        save_img(os.path.join(test_path,"test_BA_{}".format(i)), im_to_save_2)

def main(args):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print("#> Project dir : {}\n#> Current dir : {}".format(project_dir,args.save_path))
    
    # Create necessary directories
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if (not args.testing):
        # Create path
        model_path = os.path.join(args.save_path, "model")        
        train_path = os.path.join(args.save_path, "train")
        t_board_path = os.path.join(args.save_path, "logs")
        save_name = os.path.join(model_path, args.model)    

        # Create necessary directories
        if not os.path.exists(model_path): os.makedirs(model_path)
        if not os.path.exists(train_path): os.makedirs(train_path)            
        if not os.path.exists(t_board_path): os.makedirs(t_board_path)
            
        # Build the graph            
        print("#> Building graph model")
        graph = tf.Graph()
        with graph.as_default():
            cycGAN = CycleGAN()
            saver = tf.train.Saver(max_to_keep=6)

        # Load data
        data = load_data(args.dir_A, args.dir_B, file_type=args.file_type)
        
        # Start training
        with tf.Session(graph=graph) as sess:
            print("#> Training model")
            train(sess, cycGAN, data, saver, tot_epochs=args.nb_epochs,
                  save_freq=args.save_freq, test_freq=args.test_freq, log_freq=args.log_freq,
                  save_name=save_name, train_path=train_path,t_board_path=t_board_path)
            
    else:
        # Create path
        model_path = os.path.join(args.save_path, "model")  
        test_path = os.path.join(args.save_path, "test")
        save_name = os.path.join(model_path, args.model)    

        # Create necessary directories
        if not os.path.exists(model_path): raise FileNotFoundError("Model not found")            
        if not os.path.exists(test_path): os.makedirs(test_path)            
        
        # Build the graph            
        graph = tf.Graph()        
        with graph.as_default():
            cycGAN = CycleGAN(training=False)
            saver = tf.train.Saver(max_to_keep=6)
            
        # Load data
        data = load_data(args.dir_A, args.dir_B, file_type=args.file_type)

        # Start testing
        with tf.Session(graph=graph) as sess:
            print("#> Testing model")
            test( sess, cycGAN, data, saver, num_test=15, test_path=test_path, model_path=save_name)
          
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # Params for datasets
    parser.add_argument('-dA','--dir_A', dest='dir_A', default='dataset_A', help='Directory name where A images are saved')
    parser.add_argument('-dB','--dir_B', dest='dir_B', default='dataset_B', help='Directory name where B images are saved')
    # Define name save
    parser.add_argument('-m','--model', dest='model', default='CycleGAN', help='Model name')
    parser.add_argument('-s','--save_path', dest='save_path', default=os.getcwd(), help='Directory where to save all')
    parser.add_argument('-f','--file_type', dest='file_type', default='png', help='File type (png ou jpg)')
    # Params for training
    parser.add_argument('-e','--nb_epochs', dest='nb_epochs', type=int, default=100, help='Nb epochs for training')
    parser.add_argument('-sf','--save_freq', dest='save_freq', type=int, default=10, help="Saving model every 'save_freq' epochs")
    parser.add_argument('-tf','--test_freq', dest='test_freq', type=int, default=5, help="Testing model every 'test_freq' epochs")
    parser.add_argument('-lf','--log_freq', dest='log_freq', type=int, default=100, help="Displaying training log every 'log_freq' steps")
    # If defined, test a model
    parser.add_argument("--testing", help="If defined, do not train but test the model defined", action="store_true")
    args = parser.parse_args()
    main(args)
