import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as spio
#np.set_printoptions(threshold=np.inf)
from tensorflow.examples.tutorials.mnist import input_data

test_mat = spio.loadmat('frey.mat', squeeze_me=True)
#print(test_mat['ff'].shape)
frey_images=test_mat['ff']
frey_images=(frey_images-np.min(frey_images))/(np.max(frey_images)-np.min(frey_images))
w,n_samples=frey_images.shape
print('shape of mat:',frey_images.shape)
f = open('freyfaces.pkl', 'rb')
x = pickle.load(f, encoding='latin1')
f.close()
print('shape of picklw:',x.shape)
plt.figure()
plt.imshow(1-x[:1].reshape(28,20),cmap='gray')
plt.figure()
plt.imshow(frey_images[:,:1].reshape(28,20),cmap='gray')
plt.show()

def frey_next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    data=np.transpose(data)
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    return np.asarray(data_shuffle)
    #labels_shuffle = [labels[ i] for i in idx]

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


################# AutoEndoder Class#################
class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        #self.my_sess=tf.InteractiveSession()

        # Launch the session
        self._create_network()

        self.sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        '''
        x_input=self.sess.run(self.x,feed_dict={self.x: np.transpose(frey_images[:,1:2])})
        print('x_input: ',x_input)
        layer1=self.sess.run(self.layer_1,feed_dict={self.x: np.transpose(frey_images[:,1:2])})
        print('layer1: ',layer1)
        layer2=self.sess.run(self.layer_2,feed_dict={self.x: np.transpose(frey_images[:,1:2])})
        print('layer2: ',layer2)
        z_mean_print=self.sess.run(self.z_mean,feed_dict={self.x: np.transpose(frey_images[:,1:2])})
        print('zmean: ',z_mean_print)
        z_log_print=self.sess.run(self.z_log_sigma_sq ,feed_dict={self.x: np.transpose(frey_images[:,1:2])})
        print('z_log_sigma: ',z_log_print )
        z_samples=self.sess.run((self.z) ,feed_dict={self.x: np.transpose(frey_images[:,1:2])})
        print('z_samples: ',z_samples )
        '''

        # Define loss function based variational upper-bound and
        # corresponding optimizer

        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)


        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        self.eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), self.eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
        #tf.global_variables_initializer()


    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        self.layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        self.layer_2 = self.transfer_fct(tf.add(tf.matmul(self.layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = self.transfer_fct(tf.add(tf.matmul(self.layer_2, weights['out_mean']),
                        biases['out_mean']))
        z_log_sigma_sq = \
            self.transfer_fct(tf.add(tf.matmul(self.layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma']))
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

        #batch_xs = frey_next_batch(batch_size, frey_images)
    ################# Train Function #################

    def sampler(self,m,n):
         self.m=m
         self.n=n

    def enter_value(self):
        print(self.m,self.n)




network_architecture = \
    dict(n_hidden_recog_1=200, # 1st layer encoder neurons
         n_hidden_recog_2=200, # 2nd layer encoder neurons
         n_hidden_gener_1=200, # 1st layer decoder neurons
         n_hidden_gener_2=200, # 2nd layer decoder neurons
         n_input=560, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space
learning_rate=0.001
#vae = VariationalAutoencoder(network_architecture,learning_rate=learning_rate,batch_size=1)
#print(np.transpose(frey_images[:,1:2]).shape)


