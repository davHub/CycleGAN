import tensorflow as tf


class NeuralNetwork():
    def __init__(self, name='neural_net'):
        self.name = name
        
    def forward(self, inputs, reuse=False):
        pass

    def predict(self):
        pass


class PatchGAN(NeuralNetwork):
    def __init__(self, name="patchGAN"):
        NeuralNetwork.__init__(self, name=name)
        self.name = name
        self.weight_init = tf.truncated_normal_initializer(stddev=0.02)
        

    def C(self, inputs, k, use_bn=False, reuse=False, name="c"):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs, k, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=self.weight_init)
            if use_bn:
                x = tf.layers.batch_normalization(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)
            return x

    def forward(self, inputs, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            x = self.C(inputs, 64, reuse=reuse, name="c1")
            x = self.C(x, 128, use_bn=True, reuse=reuse, name="c2")
            x = self.C(x, 256, use_bn=True, reuse=reuse, name="c3")
            x = self.C(x, 512, use_bn=True, reuse=reuse, name="c4")
            x = tf.layers.conv2d(x, 1, kernel_size=4, strides=(1, 1), padding='valid', kernel_initializer=self.weight_init, name="conv_out")
            return tf.nn.sigmoid(x)


class Generator(NeuralNetwork):
    def __init__(self, nb_res_block=6, name="generator"):
        NeuralNetwork.__init__(self, name=name)
        self.name = name
        self.nb_res_block = nb_res_block
        self.weight_init = tf.truncated_normal_initializer(stddev=0.02)
        
    def res_block(self, inputs, k, index):
        with tf.variable_scope("res_block_{}_{}".format(k, index)):
            x = tf.layers.conv2d(inputs, k, kernel_size=3, strides=(1, 1), padding='SAME', kernel_initializer=self.weight_init)
            x = tf.layers.conv2d(x, k, kernel_size=3, strides=(1, 1), padding='SAME', kernel_initializer=self.weight_init)
            return (x+inputs)

    def c7s1(self, inputs, k):
        x = tf.layers.conv2d(inputs, k, kernel_size=7, strides=(1, 1), padding='SAME', kernel_initializer=self.weight_init)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        return x

    def d(self, inputs, k):
        x = tf.pad(inputs, [[0,0],[3,3],[3,3],[0,0]], "REFLECT")
        x = tf.layers.conv2d(x, k, kernel_size=3, strides=(2, 2), kernel_initializer=self.weight_init)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        return x

    def u(self, inputs, k):
        x = tf.layers.conv2d_transpose(inputs, k, kernel_size=3, strides=(2, 2), padding='SAME', kernel_initializer=self.weight_init)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        return x

    def forward(self, inputs, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            # Encoding
            x = self.c7s1(inputs, 64)
            x = self.d(x, 128)
            x = self.d(x, 256)

            # Transforming
            for ind in range(self.nb_res_block):
                x = self.res_block(x, 256, ind)

            # Decoding
            x = self.u(x, 128)
            x = self.u(x, 64)
            x = self.c7s1(inputs, 3)
            return x


class CycleGAN():
    def __init__(self, img_shape=[128, 128, 3], learning_rate=0.0002, batch_size=1, color_reg=False):
        self.lambda_adv = 1.
        self.lambda_cyc = 10.
        self.lambda_color = 0.5*self.lambda_cyc
        
        self.lr = learning_rate
        self.batch_size = batch_size

        self.A_real = tf.compat.v1.placeholder(tf.float32, shape=[None]+img_shape)
        self.B_real = tf.compat.v1.placeholder(tf.float32, shape=[None]+img_shape)

        self.gen_AtoB = Generator(name="genAB")
        self.gen_BtoA = Generator(name="genBA")

        self.dis_A = PatchGAN(name="disA")
        self.dis_B = PatchGAN(name="disB")

        dis_A_real = self.dis_A.forward(self.A_real, reuse=False)
        
        # Outputs of the generators
        self.B_fake = self.gen_AtoB.forward(self.A_real, reuse=False)
        self.A_fake = self.gen_BtoA.forward(self.B_real, reuse=False)
        
        # Outputs of the discriminator A for each type of input (real and fake image)
        dis_A_fake = self.dis_A.forward(self.A_fake, reuse=True)

        # Outputs of the discriminator B for each type of input (real and fake image)
        dis_B_real = self.dis_B.forward(self.B_real, reuse=False)
        dis_B_fake = self.dis_B.forward(self.B_fake, reuse=True)

        # Cyclic generation 
        A_cyc = self.gen_BtoA.forward(self.B_fake, reuse=True)
        B_cyc = self.gen_AtoB.forward(self.A_fake, reuse=True)

        def reduce_mean_squared(x):
            return tf.reduce_mean(tf.multiply(x,x))
            
        # Discriminator loss similar to the paper --> E_y{ (1 - D(y))^2 } + E_x{ D(G(x))^2 }
        self.dis_A_loss = reduce_mean_squared(1.-dis_A_real) + reduce_mean_squared(dis_A_fake)
        self.dis_B_loss = reduce_mean_squared(1.-dis_B_real) + reduce_mean_squared(dis_B_fake)
            
        self.dis_loss = self.dis_A_loss + self.dis_B_loss
        
        # Adversarial loss of generator similar to the paper --> E_x{ (1 - D(G(x)))^2 }
        self.adv_AB_loss = reduce_mean_squared(1.-dis_B_fake)  # + EPS sur fake dis ? 
        self.adv_BA_loss = reduce_mean_squared(1.-dis_A_fake)  # + EPS sur fake dis ?

        self.adv_loss = self.adv_AB_loss + self.adv_BA_loss

        # Cycle consistency loss
        self.cyc_loss = tf.reduce_mean(tf.abs(B_cyc - self.B_real)) + tf.reduce_mean(tf.abs(A_cyc - self.A_real))

        self.gen_loss = self.lambda_adv * self.adv_loss + self.lambda_cyc*self.cyc_loss
        
        if color_reg:
            self.color_loss = tf.reduce_mean(tf.abs(self.A_fake - self.B_real)) + tf.reduce_mean(tf.abs(self.B_fake - self.A_real))
            self.gen_loss += self.lambda_color * self.color_loss

        # We collect trainable variables
        t_vars = tf.trainable_variables()
        # dis_vars = [var for var in t_vars if 'dis' in var.name]
        gen_vars = [var for var in t_vars if 'gen' in var.name]
        
        dis_A_vars = [var for var in t_vars if 'disA' in var.name]
        dis_B_vars = [var for var in t_vars if 'disB' in var.name]
        # gen_AB_vars = [var for var in t_vars if 'genAB' in var.name]
        # gen_BA_vars = [var for var in t_vars if 'genBA' in var.name]
        
        # Optimizer
        self.dis_A_optim = tf.train.AdamOptimizer(self.lr).minimize(self.dis_A_loss, var_list=dis_A_vars)
        self.dis_B_optim = tf.train.AdamOptimizer(self.lr).minimize(self.dis_B_loss, var_list=dis_B_vars)
        self.gen_optim = tf.train.AdamOptimizer(self.lr).minimize(self.gen_loss, var_list=gen_vars)
        
    def decrease_lr(self, x):
        self.lr = self.lr - x