import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as spio
import pickle
#from tensorflow.examples.tutorials.mnist import input_data


# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py



#np.random.seed(0)
tf.set_random_seed(0)
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
rawData = spio.loadmat('/media/mehdi/NewFiles1/Task_data/AOD1.mat', squeeze_me=True)
rawData=rawData['ODD']
raw_min=np.min(rawData)
raw_max=np.max(rawData)
#test_mat=(rawData-np.min(rawData))/(np.max(rawData)-np.min(rawData))
''''
test_mat=(rawData-raw_min)/(raw_max-raw_min)
n_samples=test_mat.shape[0]
print('test_mat',test_mat[0])
'''

def frey_next_batch(num, data):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    return np.asarray(data_shuffle)
    #labels_shuffle = [labels[ i] for i in idx]

'''f = open('freyfaces.pkl', 'rb')
frey_images = pickle.load(f, encoding='latin1')
frey_images=np.transpose(frey_images)
w,n_samples=frey_images.shape
f.close()'''

#print(w,n_samples)
#print('n_samples=',n_samples)
#n_samples = mnist.train.num_examples

#################Initialization###################
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

        #if train_level==1, we are in training stage else we are testing
        #self.train_level=network_architecture["train_level"]

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        self.my_sess=tf.InteractiveSession()
        self._create_network()
        # Define loss function based variational upper-bound and
        # corresponding optimizer

        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.first_layer_w=self.sess.run(self.first_layer_weights)
        self.second_layer_w=self.sess.run(self.second_layer_weights)
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

        #print('self.batch_size2',self.batch_size)

        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)

        '''
        eps = tf.random_normal((self.x.get_shape()[0], n_z), 0, 1,
                               dtype=tf.float32)
        '''
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        (self.x_reconstr_mean,self.x_reconstr_log_sigma) = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
        #print(self.my_sess.run(self.x_reconstr_mean,feed_dict={self.x: frey_next_batch(1,test_mat)}))


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
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        self.first_layer_weights=weights['h1']
        self.second_layer_weights=weights['h2']

        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        self.layer_1_weight=weights['h1']
        self.layer_1_biases=biases['b1']

        self.layer_2_weight=weights['h2']
        self.layer_2_biases=biases['b2']

        self.weights_out_mean=weights['out_mean']
        self.weights_out_biases=biases['out_mean']


        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        x_reconstr_log_sigma = \
            tf.sigmoid(tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                    biases['out_log_sigma']))
        self.reconstr_show_mean=x_reconstr_mean;
        #self.x_reconstr_log_sigma = x_reconstr_log_sigma

        return (x_reconstr_mean,x_reconstr_log_sigma)

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)


        '''reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)'''
        reconstr_loss = \
            -tf.reduce_sum(-0.5*tf.log(2*np.pi)-0.5*(self.x_reconstr_log_sigma)-tf.divide(
                tf.pow((self.x-self.x_reconstr_mean),2),2*tf.exp(self.x_reconstr_log_sigma))
                           ,1)


        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        self.reconstr_loss=reconstr_loss
        self.latent_loss=latent_loss
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        #tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost,my_reconst,first_layer_weights= self.sess.run((self.optimizer, self.cost,self.reconstr_show_mean,self.first_layer_weights),
                                  feed_dict={self.x: X})
        #self.reconstr_show_mean=self.sess.run(self.reconstr_show_mean,feed_dict={self.x: X})
        #reconstr_show_mean,x_reconstr_log_sigma=self.sess.run((self.reconstr_show_mean,self.x_reconstr_log_sigma),feed_dict={self.x:X})
        #print('layer_1_eval',layer_1_eval)
        reconstr_loss,latent_loss=self.sess.run((self.reconstr_loss,self.latent_loss),feed_dict={self.x: X})

        #print('reconstr_loss:',reconstr_loss,'\n latent_loss: ',latent_loss)
        #print('\n')
        #print('x_reconstr_log_sigma:\n',x_reconstr_log_sigma)

        return (cost,my_reconst,first_layer_weights)

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        my_reconstr_mean= self.sess.run(self.x_reconstr_mean,feed_dict={self.x: X})
        my_reconstr_sigma= self.sess.run(self.x_reconstr_mean,feed_dict={self.x: X})
        return (my_reconstr_mean,my_reconstr_sigma)

    def test_reconstruct(self,X):
        (z_mean, z_log_sigma_sq)=self.sess.run((self.z_mean,self.z_log_sigma_sq),feed_dict={self.x: X})
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((X.shape[0], n_z), 0, 1,
                               dtype=tf.float32)
        z = tf.add(z_mean,
                   tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, self.layer_1_weight),
                                           self.layer_1_biases))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, self.layer_2_weight),
                                           self.layer_2_biases))

        x_reconstr_mean = \
            self.sess.run((tf.sigmoid(tf.add(tf.matmul(layer_2, self.weights_out_mean),
                              self.weights_out_biases))))
        '''
        z_log_sigma_sq=self.sess.run(self.z_log_sigma_sq,feed_dict={self.x: X})
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((X.shape[0], n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        z = tf.add(z_mean,
                        tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input

        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        x_reconstr_log_sigma = \
            tf.sigmoid(tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                    biases['out_log_sigma']))
        '''
        return (z_mean,x_reconstr_mean)

################# Train Function #################
def train(data_input,network_architecture, learning_rate=0.0005,
          batch_size=20, training_epochs=10, display_step=1):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    loss_vec=np.array([])
    loss_step=np.array([])
    #plt.figure(figsize=(16, 10))
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        #plt.show(block=False)
        # Loop over all batches
        plt.ion()
        for i in range(total_batch):
            #batch_xs, _ = mnist.train.next_batch(batch_size)
            batch_xs = frey_next_batch(batch_size,data_input)
            #print(batch_xs.shape)
            # Fit training using batch data
            '''
            plt.subplot(211)
            plt.plot(batch_xs[0,:])

            #plt.subplot(212)
            #plt.plot(vae.reconstr_show_mean[0,:])

            (cost,my_reconstr,first_layer_weights) = vae.partial_fit(batch_xs)


            plt.subplot(212)
            plt.plot(my_reconstr[0,:])

            plt.show()
            plt.pause(0.005)
            plt.gcf().clear()
            '''
            (cost,my_reconstr,first_layer_weights) = vae.partial_fit(batch_xs)
            #print('this cost:',cost)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size
        # Display logs per epoch step
            #print(vae.reconstr_show_mean)

        loss_step = np.append(loss_step,epoch)
        loss_vec = np.append(loss_vec,avg_cost)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    #plt.plot(loss_step,loss_vec)
    #plt.show()
    return vae,loss_step,loss_vec


################# Train and Test Function ###############################################################

###Training percent, control and patient sizes
Train_size=0.7
control_size=150
patient_size=121

###Extracting subsamples from control and patient
Con_samp=int(round(Train_size*control_size))
Pat_samp=int(round(Train_size*patient_size))

###Indexes of training and testing
ID_con=np.random.permutation(control_size)
ID_pat=control_size+np.random.permutation(patient_size)

Index_con_train = ID_con[0:Con_samp]
Index_con_test = ID_con[Con_samp+0:control_size]

Index_pat_train = ID_pat[0:Pat_samp]
Index_pat_test = ID_pat[Pat_samp+0:patient_size]

test_mat=(rawData-raw_min)/(raw_max-raw_min)
n_samples=test_mat.shape[0]

X_train_con=test_mat[Index_con_train,:]
X_train_pat=test_mat[Index_pat_train,:]
X_train=np.concatenate((X_train_con, X_train_pat), axis=0)

X_test_con=test_mat[Index_con_test,:]
X_test_pat=test_mat[Index_pat_test,:]
X_test=np.concatenate((X_test_con, X_test_pat), axis=0)


print('X_train',X_train.shape)
print('X_test',X_test.shape)
#print('ID_con',ID_con)
print('Index_pat_train',np.sort(Index_pat_train))
print('Index_pat_test',np.sort(Index_pat_test))

#print('Index_pat_train',Index_pat_train)

#print('Index_pat_train_min',np.min(Index_pat_train))

'''
network_architecture = \
    dict(n_hidden_recog_1=200, # 1st layer encoder neurons
         n_hidden_recog_2=200, # 2nd layer encoder neurons
         n_hidden_gener_1=200, # 1st layer decoder neurons
         n_hidden_gener_2=200, # 2nd layer decoder neurons
         n_input=X_train.shape[1], # MNIST data input (img shape: 28*28)
         n_z=20,
         )  # dimensionality of latent space
batch_size=20
#(vae,loss_step,loss_vec) = train(X_train,network_architecture,batch_size=batch_size, training_epochs=1)
X_test=test_mat[0:2,:]
(z_mean,x_reconstr_mean)=vae.test_reconstruct(X_test)
print('x_reconstr_mean_shpe: ',x_reconstr_mean.shape)
plt.subplot(211)
#plt.plot(X_test[0, :])
plt.plot(X_test[0, :])
#plt.plot(x_reconstr_mean[0,:])
plt.subplot(212)
plt.plot(x_reconstr_mean[0, :])
plt.show()
plt.pause(5)

#print(z_mean)
#print(z_mean.shape)
#print(my_reconstr_mean.shape)
'''