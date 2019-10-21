import time
import os
import random
import tensorflow as tf
from collections import deque

from tools import *
from ops import instance_norm, batch_norm
from image_buff import ImageBuffer

class NeuralNetwork():
    """ Define an abstract class of neural network
    
    The neural network class is an abstract class with the methods that a NN architecture should implement.
    """
    def __init__(self, name='neural_net'):
        """ Initialize a NN class
        
        Args:
            name: the name of the NN
        """
        self.name = name
        
    def forward(self, inputs, reuse=False):
        """ Define the forward path in the NN
        
        Args:
            inputs: the inputs of the block
            reuse: reuse or not the same layers parameters
        """
        pass


class PatchGAN(NeuralNetwork):
    """ Define a PatchGAN architecture
    
    The PatchGAN is an architecture built to be a good discriminator in GAN.
    """
    def __init__(self, name="patchGAN", norm="instance"):
        """ Initialize a PatchGAN class
        
        Args:
            name: the name of the NN
        """
        NeuralNetwork.__init__(self, name=name)
        
        if norm=="batch":
            self.norm = batch_norm
        else :
            self.norm = instance_norm

    def C(self, inputs, k, use_bn=True, name="c"):
        """ Define the forward path in the NN
        
        Args:
            inputs: the inputs of the block
            k: the number of filters
            use_bn: use batch normalization or not
            reuse: reuse or not the same layers parameters
            name: the name of the block            
        """
        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs, k, kernel_size=4, strides=(2, 2), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            if use_bn:
                x = self.norm(x, name='batch_norm')
            x = tf.nn.leaky_relu(x, alpha=0.2)
            return x

    def forward(self, inputs, reuse=True):
        """ Define the forward path in the NN
        
        Args:
            inputs: the inputs of the block
            reuse: reuse or not the same layers parameters
        """
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
                
            x = self.C(inputs, 64, use_bn=False, name="c1")
            x = self.C(x, 128, name="c2")
            x = self.C(x, 256, name="c3")
            x = self.C(x, 512, name="c4")
            x = tf.layers.conv2d(x, 1, kernel_size=4, strides=(1, 1), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name="conv_out")
            return x

class Generator(NeuralNetwork):
    """ Define the generator
    
    The generator is an architecture adopted from Johnson and al. architecture.
    """
    def __init__(self, nb_res_block=6, name="generator", dropout=None, norm="instance"):
        """ Initialize a generator class
        
        Args:
            nb_res_block: the number of residual block in the architecture
            name: the name of the NN
            dropout: the amount of dropout to apply
        """
        NeuralNetwork.__init__(self, name=name)
        self.nb_res_block = nb_res_block
        self.dropout = dropout
        
        if norm=="batch":
            self.norm = batch_norm
        else :
            self.norm = instance_norm
       
    # Resnet Block 
    def res_block(self, inputs, k, name="res_block", dropout=None):
        """ Define a residual block
        
        Args:
            inputs: the inputs of the block
            k: the number of filters
            dropout: the amount of dropout to apply
        """
        with tf.variable_scope(name):
            x = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")            
            x = tf.layers.conv2d(x, k, kernel_size=3, strides=(1, 1), padding='VALID', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv1')
            x = self.norm(x, name="bn1")
            x = tf.nn.relu(x)
            if dropout is not None:
                x = tf.nn.dropout(x, dropout)
            x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
            x = tf.layers.conv2d(x, k, kernel_size=3, strides=(1, 1), padding='VALID', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
            x = self.norm(x, name="bn2")
            return (x+inputs)


    def c7s1(self, inputs, k, use_bn=True, activ=tf.nn.relu, name='c7s1'):
        """ Define a convolutional block (same notations than in the paper)
        
        Args:
            inputs: the inputs of the block
            k: the number of filters
            use_bn: use batch normalization or not
            activ: activation function to apply (default: ReLU)
        """
        with tf.variable_scope(name):
            x = tf.pad(inputs, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
            x = tf.layers.conv2d(x, k, kernel_size=7, strides=(1, 1), padding="VALID", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            if use_bn:
                x = self.norm(x, name='batch_norm')
            x = activ(x)
            return x

    # Downsampling layers
    def downsampling(self, inputs, k, name='d'):
        """ Define a downsampling block (d_k in paper notations)
        
        Args:
            inputs: the inputs of the block
            k: the number of filters
        """
        with tf.variable_scope(name):
            x = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]])        
            x = tf.layers.conv2d(x, k, kernel_size=3, strides=(2, 2), kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            x = self.norm(x, name='norm')
            return tf.nn.relu(x)

    # Upsampling layers
    def upsampling(self, inputs, k, name='u'):
        """ Define an upsampling block (u_k in paper notations)
        
        Args:
            inputs: the inputs of the block
            k: the number of filters
        """
        with tf.variable_scope(name):
            x = tf.layers.conv2d_transpose(inputs, k, kernel_size=3, strides=(2, 2), padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            x = self.norm(x, name='norm')
            x = tf.nn.relu(x)
            return x

    def forward(self, inputs, reuse=True):
        """ Define the forward path in the NN
        
        Args:
            inputs: the inputs of the block
            reuse: reuse or not the same layers parameters
        """
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
                
            # Encoding
            x = self.c7s1(inputs, 64, name='c7s1_64')
            x = self.downsampling(x, 128, name='d128')
            x = self.downsampling(x, 256, name='d256')

            # Transforming
            for ind in range(self.nb_res_block):
                x = self.res_block(x, 256, name='res_block_{}'.format(ind), dropout=self.dropout)

            # Decoding
            x = self.upsampling(x, 128, name='u128')
            x = self.upsampling(x, 64, name='u64')
            x = self.c7s1(inputs, 3, use_bn=False, activ=tf.nn.tanh)
            return x


class CycleGAN():
    """ Define CycleGAN model.
    
    The CycleGAN model is a model from the paper Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.
    """
    def __init__(self, img_shape=[128, 128, 3], learning_rate=0.0002, beta1=0.9, norm='instance', color_reg=False, testing=False):
        """ Initialize a CycleGAN class.
        
        Args:
            img_shape (list): the shape of images
            learning_rate: the learning rate
            beta1: parameter for optimizer
            color_reg (bool): if True, add color regularization
            training: if True, construct training model
        """
        self.lambda_adv = 1.
        self.lambda_cyc = 10.
        self.lambda_color = 0.5*self.lambda_cyc
        
        self.img_shape = img_shape
        self.lr = learning_rate
        self.beta1 = beta1

        self.A_real = tf.compat.v1.placeholder(tf.float32, shape=[None]+img_shape, name="pl_A_real")
        self.B_real = tf.compat.v1.placeholder(tf.float32, shape=[None]+img_shape, name="pl_B_real")
        
        self.lr_pl = tf.placeholder(tf.float32, None, name='learning_rate')
        
        self.A_fake_buff = tf.compat.v1.placeholder(tf.float32, shape=[None]+img_shape, name="pl_A_fake")
        self.B_fake_buff = tf.compat.v1.placeholder(tf.float32, shape=[None]+img_shape, name="pl_B_fake")

        # Create generators
        nb_res_block= 9 if img_shape[0]>200 else 6
        self.genA2B = Generator(nb_res_block=nb_res_block, norm=norm, name="genA2B")
        self.genB2A = Generator(nb_res_block=nb_res_block, norm=norm, name="genB2A")

        # Outputs of the generators
        self.B_fake = self.genA2B.forward(self.A_real, reuse=False)
        self.A_fake = self.genB2A.forward(self.B_real, reuse=False)      

        # Cyclic generation
        self.A_cyc = self.genB2A.forward(self.B_fake, reuse=True)
        self.B_cyc = self.genA2B.forward(self.A_fake, reuse=True)

        if not testing:
            self.buffer_fake_A = ImageBuffer(maxlen=50)
            self.buffer_fake_B = ImageBuffer(maxlen=50)
            
            # Create and display discriminators
            self.dis_A = PatchGAN(norm=norm, name="disA")
            self.dis_B = PatchGAN(norm=norm, name="disB")

            # Outputs of the discriminators for real images
            self.dis_A_real = self.dis_A.forward(self.A_real, reuse=False)
            self.dis_B_real = self.dis_B.forward(self.B_real, reuse=False)

            # Outputs of the discriminators for fake image
            self.dis_A_fake = self.dis_A.forward(self.A_fake, reuse=True)
            self.dis_A_fake_buff = self.dis_A.forward(self.A_fake_buff, reuse=True)
            
            self.dis_B_fake = self.dis_B.forward(self.B_fake, reuse=True)
            self.dis_B_fake_buff = self.dis_B.forward(self.B_fake_buff, reuse=True)

            # def adv_loss(logits, labels):
            #   return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

            def adv_loss(values, target):
                return tf.reduce_mean((values-target)**2)

            def abs_loss(values, target):
                return tf.reduce_mean(tf.abs(values - target))
            
            # Discriminator loss similar to the paper --> E_y{ (1 - D(y))^2 } + E_x{ D(G(x))^2 }
            # factor 0.5 to slow down D learning
            self.dis_A_loss = 0.5*(adv_loss(self.dis_A_real, tf.ones_like(self.dis_A_real) + adv_loss(self.dis_A_fake_buff, tf.zeros_like(self.dis_A_fake_buff)))) #reduce_mean_squared(1.-dis_A_real) + reduce_mean_squared(dis_A_fake_buff))
            self.dis_B_loss = 0.5*(adv_loss(self.dis_B_real, tf.ones_like(self.dis_B_real) + adv_loss(self.dis_B_fake_buff, tf.zeros_like(self.dis_B_fake_buff)))) #reduce_mean_squared(1.-dis_A_real) + reduce_mean_squared(dis_A_fake_buff))

            self.dis_loss = self.dis_A_loss + self.dis_B_loss

            # Adversarial loss of generator similar to the paper --> E_x{ (1 - D(G(x)))^2 }
            self.adv_AB_loss = adv_loss(self.dis_B_fake, tf.ones_like(self.dis_B_fake))
            self.adv_BA_loss = adv_loss(self.dis_A_fake, tf.ones_like(self.dis_A_fake))

            self.adv_loss = self.adv_AB_loss + self.adv_BA_loss

            # Cycle consistency loss
            self.cyc_loss = abs_loss(self.B_cyc, self.B_real) + abs_loss(self.A_cyc, self.A_real)

            self.gen_loss = self.lambda_adv * self.adv_loss + self.lambda_cyc*self.cyc_loss

            # Add color regularization if needed
            if color_reg:
                self.color_loss = abs_loss(self.A_fake, self.B_real) + abs_loss(self.B_fake, self.A_real)
                self.gen_loss += self.lambda_color * self.color_loss

            # We collect trainable variables
            t_vars = tf.trainable_variables()
            self.dis_vars = [var for var in t_vars if 'dis' in var.name]
            self.gen_vars = [var for var in t_vars if 'gen' in var.name]
            
            # Display model
            for var in t_vars: 
                print(var.name)

            # Optimizer
            self.dis_optim = tf.train.AdamOptimizer(self.lr_pl, beta1=self.beta1).minimize(self.dis_loss, var_list=self.dis_vars)
            self.gen_optim = tf.train.AdamOptimizer(self.lr_pl, beta1=self.beta1).minimize(self.gen_loss, var_list=self.gen_vars)
        
    def decrease_lr(self, x):
        """ Decrease the learning rate
        
        Args:
            x: the quantity to substract
        """
        self.lr = min(self.lr - x,0)
        
    def train(self, sess, train_data, saver, tot_epochs=200, save_freq=10, test_freq=5, log_freq=20, save_name="model", train_path="./",t_board_path="./", model_path=None):
        """ Train a model
    
        Args:
            sess: tensorflow session
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
        if num_batches==0:
            raise ValueError()
        print("#> Number of batches {}".format(num_batches))

        # Get image to test and analyse training 
        n_test=8
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
                decay = self.lr/(tot_epochs//2)
                self.decrease_lr(decay)

            for step in range(num_batches):
                # Get next batch
                train_A = make_noisy(train_data['A'][step], crop_size=self.img_shape[0])
                train_B = make_noisy(train_data['B'][step], crop_size=self.img_shape[0])

                if len(train_A.shape)<4: train_A = [train_A]
                if len(train_B.shape)<4: train_B = [train_B]

                # We update generators params
                _, gen_loss, A_fake, B_fake = sess.run([self.gen_optim, self.gen_loss, self.A_fake, self.B_fake], feed_dict={ self.A_real: train_A, self.B_real: train_B, self.lr_pl: self.lr })

                # Update discriminators params
                _, dis_loss = sess.run([self.dis_optim, self.dis_loss], feed_dict={   self.A_real: train_A, self.B_real: train_B,
                                                                                        self.A_fake_buff: self.buffer_fake_A(A_fake), self.B_fake_buff: self.buffer_fake_B(B_fake),
                                                                                        self.lr_pl: self.lr})

                # Display log every 'log_freq' steps  
                if (step+1) % log_freq == 0:
                    time_now = time.time()
                    t_since_start = (time_now-start_time)/60
                    print("#> Ep {} Step {}/{} - ({:3.4}min) -- Losses dis({:.6})  gen({:.6})".format(epoch, step+1, num_batches,t_since_start, dis_loss, gen_loss))

            # Test model every 'test_freq' epochs  
            if (epoch+1) % test_freq == 0:
                print('#> Testing')
                gen_AB, gen_BA = sess.run([self.B_fake, self.A_fake], feed_dict={ self.A_real: test_A, self.B_real: test_B })
                save_img(os.path.join(train_path,"genAB_ep{}".format(epoch)), group_images(deprocess(gen_AB)))
                save_img(os.path.join(train_path,"genBA_ep{}".format(epoch)), group_images(deprocess(gen_BA)))

            # Save model every 'save_freq' epochs  
            if (epoch+1) % save_freq == 0:
                print("#> Saving self")
                saver.save(sess, save_name, global_step=global_step)

            global_step = global_step + 1   
            
    def test(self, sess, test_data, saver, num_test=10, test_path="./", model_path=None):
        """ Test a model
    
        Args:
            sess: tensorflow session
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

        A_fake, B_fake, A_cyc, B_cyc = sess.run([self.A_fake, self.B_fake, self.A_cyc, self.B_cyc], feed_dict={ self.A_real: test_A, self.B_real: test_B })

        for i in range(num_test):
            im_to_save_1 = np.concatenate((deprocess(test_A[i]), deprocess(B_fake[i]), deprocess(A_cyc[i])), axis=1)
            save_img(os.path.join(test_path,"test_AB_{}".format(i)), im_to_save_1)
            im_to_save_2 = np.concatenate((deprocess(test_B[i]), deprocess(A_fake[i]), deprocess(B_cyc[i])), axis=1)
            save_img(os.path.join(test_path,"test_BA_{}".format(i)), im_to_save_2)
