'''
routes - binary vector of ones and zeros with vectorized beta-binomial distribution
active - positive numbers with normal-normal distribution

Steps:
 1. Get data from routes and active
 2. Initialize alpha, beta, theta from beta-binomial and mu, sigma^2, mu_0, sigma_0, nu, eta
3a. Draw n1 samples from routes
3b. Use route samples to update normal-normal parameters for m1 steps
3c. Get n2 predictive posterior samples of actives
3d. Use active samples to update beta-binomial parameters for m2 steps
3e. Draw n1 predictive posterior samples of routes
3f. Repeat  b - e, k times
 4. Update W parameter using gradient descent/ascent (??) from binomial likelihood
5a. Draw n2 samples from actives
5b. Use active samples to update beta-binomial model parameters for m2 steps
5c. Get n1 predictive posterior samples of routes
5d. Use route samples to update normal-normal parameters for m1 steps
5e. Draw n2 predictive posterior samples of actives
5f. Repeat b - e, k times
 6. Update W parameter using gradient descent/ascent (??) from normal likelihood
 7. Repeat 3 - 6 EPOCHS number of times

To start with the simulation can just simulate step 1

Much of this code is based on information from the following sources:
http://lyy1994.github.io/machine-learning/2017/04/17/RBM-tensorflow-implementation.html
http://deeplearning.net/tutorial/rbm.html
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import normalize
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

#tf.enable_eager_execution()

tfd = tfp.distributions

class Data:
    '''
    I want this to be able to get the real data and simulate data
    '''
    def __init__(self):
        pass

    def simulate(self, num_positions=3, num_samples=300, p_vector=(.4, .6), active_mu = (10, 15, 20, 25), active_sigma = (2, 1.5, 1.5, 2), success_probability=0.5, seed=635):
        '''
        :param num_routes: number of classes within multinomials
        :param num_samples: number of samples
        :param p_vector: probability vector for mulitinomials, tuple
        :param active_mu: vector of normal means for active samples
        :param active_sigma: vector of normal standard deviations for active samples
        :param success_probability: probabilty of route success
        :param seed: random seed
        :return: dictionary of route samples and active samples and success responses
        '''

        assert len(active_mu) == len(active_sigma), "number of mu's need to be equal to sigma's"
        assert 0 < sum(p_vector) <= 1, "Sum of probability vector greater than one or less than zero"

        #set seed
        np.random.seed(seed)

        num_routes = len(p_vector)
        num_active = len(active_mu)

        #routes is a matrix of dimension num_samples by (num_positions x num_routes)
        routes = np.empty([num_samples, num_positions*num_routes])
        for i in range(num_positions):
            routes[:, list(range(i*num_routes, (i+1)*num_routes))] = np.random.multinomial(1, p_vector, size=num_samples)

        for i in range(num_samples):
            assert sum(routes[i, :]) == num_positions, 'The num routes is {} and the sum is {}'.format(num_routes, sum(routes[i, :]))

        #actives is a matrix of dimension num_samples by num_active
        actives = np.empty([num_samples, num_active])
        for i in range(num_active):
            actives[:, i] = np.random.normal(loc=active_mu[i], scale=active_sigma[i], size=num_samples)

        assert np.min(actives) > 0, 'Not all members of actives are positive'

        successes = np.random.binomial(1, success_probability, num_samples)

        return {'actives': actives, 'routes': routes, 'sucesses': successes}

    def real(self):
        pass


def weight(shape, name='weights'):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

class RBM:
    #flag = True #flipping index for computing likelihood

    def __init__(self, n_route, n_position, n_active, k=3, m=10):
        '''
        :param n_route: Number of routes recorded for each position
        :param n_position: Number of spots the routes could be run from (in my data this is five)
        :param n_active: Number of active datapoints (things like speed, angle, distance, etc.)
        :param k: Number of steps in CD learning
        :param m: Number of steps in MCMC updating for normal-normal case
        '''
        self.n_route = n_route
        self.n_position = n_position
        self.n_active = n_active
        self.k = k
        self.m = m

        #learning rate
        self.lr = tf.placeholder(tf.float32) if not tf.executing_eagerly() else tf.constant(1., dtype=tf.float32)

        #weight matrix
        self.w = weight([n_active, n_position*n_route], 'W')

        #constant parameters
        self.mu_not = tf.constant([0.]*n_active, dtype=tf.float32)
        self.sigma_not = tf.constant([1.]*n_active, dtype=tf.float32)
        self.eta = tf.constant([1.]*n_active, dtype=tf.float32)
        self.nu = tf.constant([1.]*n_active, dtype=tf.float32)
        self.alpha = tf.constant([1.]*n_route, dtype=tf.float32)

        #variable parameters
        self.mu = tf.Variable(tf.truncated_normal([n_active], stddev=0.1), name='mu', dtype=tf.float32)
        self.sigma = tf.Variable(tf.truncated_normal([n_active], mean=1, stddev=0.1), name='sigma', dtype=tf.float32)
        self.theta = tf.Variable(normalize(np.random.normal(loc=.5, scale=0.01, size=[n_position, n_route]), norm='l1'), name='theta', dtype=tf.float32)


    '''
    Posterior predicitve functions. 
    The down generates active realizations based on updated variable parameters (mu and sigma) and initial route realizations.
    The up generates route realizations based on updated variable parameters (theta).
    '''
    def postpreddown(self, routes):

        #number of samples to generate
        num_samples = tf.cast(tf.shape(routes)[0], tf.float32)

        #use mean of samples
        routes_mean = tf.reshape(tf.cast(tf.reduce_mean(routes, axis=0), tf.float32), [-1, 1])

        assert routes_mean.get_shape().as_list() == [self.n_position*self.n_route, 1], 'Routes mean should have {} elements but has {}'.format(self.n_position*self.n_route, routes_mean.get_shape().as_list()[1])

        #location and scale vectors according to predictive posterior calculations
        mean_vec1 = tf.multiply(tf.divide(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.add(tf.square(self.sigma), tf.multiply(num_samples, tf.square(self.sigma_not)))), tf.reshape(tf.matmul(self.w, routes_mean), [-1])) #have to compress column vectors into arrays otherwise broadcasting gets screwy
        mean_vec2 = tf.multiply(tf.divide(tf.square(self.sigma), tf.add(tf.square(self.sigma), tf.multiply(num_samples, tf.square(self.sigma_not)))), self.mu_not)
        mean_vec = tf.add(mean_vec1, mean_vec2)

        sigma_vec = tf.sqrt(tf.add(tf.divide(tf.multiply(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma)), tf.add(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma))), tf.square(self.sigma)))

        assert mean_vec.get_shape().as_list() == self.mu.get_shape().as_list(), 'mu shapes for posterior predictive of normal-normal are different'
        assert sigma_vec.get_shape().as_list() == self.sigma.get_shape().as_list(), 'sigma shapes for posterior predictive of normal-normal are different'

        #create mulivariate normal distribution assuming independence between observations
        multi_normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)

        #should be a tensor with dimension number of samples by number of active data points
        return multi_normal.sample(sample_shape=tf.cast(num_samples, tf.int32))

    def postpredup(self, num_samples):

        #cast so sample can handle the input
        num_samples = tf.cast(num_samples, tf.int32)

        #initialize multinomial distribution drawing one route per position
        multinomial = tfd.Multinomial(total_count=tf.constant([1.]*self.n_position, dtype=tf.float32), probs=self.theta, validate_args=True)

        #draw samples and reshape to conform to structure I've been using
        sample = multinomial.sample(num_samples)
        return tf.reshape(sample, [num_samples, self.n_position*self.n_route])

    '''
    Gradient with respect to the log likelihood of a normal or multinomial distribution depending on direction
    '''
    def gradientup(self, route_samples):
        route_samples = tf.cast(tf.convert_to_tensor(route_samples), tf.float32)

        out = tf.matmul(tf.diag(tf.reciprocal(self.sigma)), tf.matmul(tf.subtract(tf.matmul(self.w, tf.transpose(route_samples)), tf.reshape(self.mu, [-1, 1])), route_samples))

        return out

    def gradientdown(self, active_samples):
        active_samples = tf.cast(tf.convert_to_tensor(active_samples), tf.float32)
        num_samples = tf.gather(tf.shape(active_samples), 0)

        out = tf.matmul(tf.transpose(active_samples), tf.broadcast_to(tf.log(tf.divide(tf.reshape(self.theta, [1, self.n_position*self.n_route]), tf.subtract(1., tf.reshape(self.theta, [1, self.n_position*self.n_route])))), tf.stack([num_samples, tf.convert_to_tensor(self.n_position*self.n_route, dtype=tf.int32)], 0)))

        return out

    '''
    Contrastive Divergence to train the weight matrix in the middle. Goes through k iterations of passing posterior
    predictives back and forth and updating the parameters of the normal-normal model. Not clear how to do this for
    the dirichlet-multinomial distribution. Also might want to look into keeping track of the chain to plot later
    to make sure that everything is mixing well.
    '''
    def CD_k_up(self, route_samples):

        in_samples = tf.cast(tf.convert_to_tensor(route_samples), tf.float32)

        #given true route samples trade back and forth and produce new route samples at the end
        num_samples_int = tf.shape(route_samples)[0]
        num_samples_float = tf.cast(num_samples_int, tf.float32)

        for _ in range(self.k):

            #use the means to compact information
            routes_mean = tf.cast(tf.reshape(tf.reduce_mean(route_samples, axis=0), [-1, 1]), tf.float32)

            for _ in range(self.m):

                #get the mean vec according to posterior of mu
                mean_vec = tf.add(tf.multiply(tf.divide(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.add(tf.square(self.sigma), tf.multiply(num_samples_float, tf.square(self.sigma_not)))), tf.reshape(tf.matmul(self.w, routes_mean), [-1])),
                                  tf.multiply(tf.divide(tf.square(self.sigma), tf.add(tf.square(self.sigma), tf.multiply(num_samples_float, tf.square(self.sigma_not)))), self.mu_not))

                #get the sigma vec according to the posterior of mu
                sigma_vec = tf.sqrt(tf.divide(tf.multiply(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.square(self.sigma)),
                                              tf.add(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.square(self.sigma))))

                assert mean_vec.get_shape().as_list() == self.mu.get_shape().as_list(), 'Mean vector should have same shape as mu in normal-normal'
                assert sigma_vec.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Scale vector should have same shape as sigma in normal-normal'

                #intialize and draw new mu
                normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)
                _ = tf.assign(self.mu, normal.sample())

                #get alpha and beta according to posterior of sigma
                concentration = tf.add(self.nu, 0.5)
                rate = tf.divide(tf.add(tf.multiply(2., self.nu), tf.square(tf.subtract(tf.reshape(tf.matmul(self.w, routes_mean), [-1]), self.mu))), 2.)

                assert concentration.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Alpha of inverse gamma should have as many elements as sigma'
                assert rate.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Beta of inverse gamma should have as many elements as sigma'

                #draw sigma vector indpendently since I'm assuming independence right now
                inv_gamma = tfd.InverseGamma(concentration=concentration, rate=rate)
                _ = tf.assign(self.sigma, inv_gamma.sample())

            #draw new active samples, within function uses updated mu and sigma
            active_samples = self.postpreddown(route_samples)

            #work with mean reduction of data
            actives_mean = tf.cast(tf.reshape(tf.reduce_mean(active_samples, axis=0), [1, -1]), tf.float32)

            #initialize alphas of the dirichlet
            new_alphas = tf.maximum(tf.matmul(actives_mean, self.w), tf.cast(tf.Variable(-1 * tf.reduce_max(self.alpha) + 1e-4), tf.float32))

            assert new_alphas.get_shape().as_list()[0] == 1, 'New alphas should be a single row'
            assert new_alphas.get_shape().as_list()[1] == self.n_route*self.n_position, 'New alphas should have a spot for each route/position combination'

            #draw thetas from dirichlet
            new_thetas = []
            for i in range(self.n_position):
                #add prior alphas to new alphas and draw for each position
                dirichlet = tfd.Dirichlet(concentration=tf.add(self.alpha, new_alphas[0, i*self.n_route:(i+1)*self.n_route]))
                new_thetas.append(dirichlet.sample())

            #reshape flattened thetas to fit into n_positions by n_routes shape
            _ = tf.assign(self.theta, tf.reshape(tf.convert_to_tensor(new_thetas), tf.shape(self.theta)))

            #use updated theta to sample routes
            route_samples = self.postpredup(num_samples=num_samples_int)

        out_samples = route_samples

        assert out_samples.get_shape().as_list() == in_samples.get_shape().as_list(), 'Samples should have the same shape in CD_k_up'

        gradient = tf.subtract(self.gradientup(in_samples), self.gradientup(out_samples))

        assert gradient.get_shape().as_list() == self.w.get_shape().as_list(), 'Gradient should have the same shape as weight matrix'

        return gradient

    def CD_k_down(self, active_samples):

        in_samples = tf.cast(tf.convert_to_tensor(active_samples), tf.float32)

        # given true active samples trade back and forth and produce new active samples at the end
        num_samples_int = tf.shape(active_samples)[0]
        num_samples_float = tf.cast(num_samples_int, tf.float32)

        #see comments above, same but in different order
        for _ in range(self.k):
            #work with mean reduction of data
            actives_mean = tf.cast(tf.reshape(tf.reduce_mean(active_samples, axis=0), [1, -1]), tf.float32)

            # initialize alphas of the dirichlet
            new_alphas = tf.maximum(tf.matmul(actives_mean, self.w), tf.cast(tf.Variable(-1 * tf.reduce_max(self.alpha) + 1e-4), tf.float32))

            assert new_alphas.get_shape().as_list()[0] == 1, 'New alphas should be a single row'
            assert new_alphas.get_shape().as_list()[1] == self.n_route*self.n_position, 'New alphas should have a spot for each route/position combination'

            #draw thetas from dirichlet
            new_thetas = []
            for i in range(self.n_position):
                #add prior alphas to new alphas and draw for each position
                dirichlet = tfd.Dirichlet(concentration=tf.add(self.alpha, new_alphas[0, i*self.n_route:(i+1)*self.n_route]))
                new_thetas.append(dirichlet.sample())

            #reshape flattened thetas to fit into n_positions by n_routes shape
            _ = tf.assign(self.theta, tf.reshape(tf.convert_to_tensor(new_thetas), tf.shape(self.theta)))

            #use updated theta to sample routes
            route_samples = self.postpredup(num_samples=num_samples_int)

            #use the means to compact information
            routes_mean = tf.cast(tf.reshape(tf.reduce_mean(route_samples, axis=0), [-1, 1]), tf.float32)

            for _ in range(self.m):

                #get the mean vec according to posterior of mu
                mean_vec = tf.add(tf.multiply(tf.divide(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.add(tf.square(self.sigma), tf.multiply(num_samples_float, tf.square(self.sigma_not)))), tf.reshape(tf.matmul(self.w, routes_mean), [-1])),
                                  tf.multiply(tf.divide(tf.square(self.sigma), tf.add(tf.square(self.sigma), tf.multiply(num_samples_float, tf.square(self.sigma_not)))), self.mu_not))

                #get the sigma vec according to the posterior of mu
                sigma_vec = tf.sqrt(tf.divide(tf.multiply(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.square(self.sigma)),
                                              tf.add(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.square(self.sigma))))

                assert mean_vec.get_shape().as_list() == self.mu.get_shape().as_list(), 'Mean vector should have same shape as mu in normal-normal'
                assert sigma_vec.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Scale vector should have same shape as sigma in normal-normal'

                #intialize and draw new mu
                normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)
                _ = tf.assign(self.mu, normal.sample())

                #get alpha and beta according to posterior of sigma
                concentration = tf.add(self.nu, 0.5)
                rate = tf.divide(tf.add(tf.multiply(2., self.nu), tf.square(tf.subtract(tf.reshape(tf.matmul(self.w, routes_mean), [-1]), self.mu))), 2.)

                assert concentration.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Alpha of inverse gamma should have as many elements as sigma'
                assert rate.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Beta of inverse gamma should have as many elements as sigma'

                #draw sigma vector indpendently since I'm assuming independence right now
                inv_gamma = tfd.InverseGamma(concentration=concentration, rate=rate)
                _ = tf.assign(self.sigma, inv_gamma.sample())

            #draw new active samples, within function uses updated mu and sigma
            active_samples = self.postpreddown(route_samples)

        #after pinballing get last set of samples to train weight matrix not to output
        out_samples = active_samples

        assert out_samples.get_shape().as_list() == in_samples.get_shape().as_list(), 'Samples should have the same shape in CD_k_down'

        #positive is true sample, negative is generated sample
        gradient = tf.subtract(self.gradientdown(in_samples), self.gradientdown(out_samples))

        assert gradient.get_shape().as_list() == self.w.get_shape().as_list(), 'Gradient should have the same shape as weight matrix'

        return gradient

    def learn(self, route_samples, active_samples):
        if tf.executing_eagerly():
            assert route_samples.get_shape().as_list()[0] > 0, 'Need at least one route sample'
            assert active_samples.get_shape().as_list()[0] > 0, 'Need at least one active sample'

            assert route_samples.get_shape().as_list()[1] == self.n_position * self.n_route, 'Number of observations in the route sample is not correct'
            assert active_samples.get_shape().as_list()[1] == self.n_active, 'Number of observations in the active sample is not correct'

        #get the gradient and update the weight matrix
        w_grad_up = self.CD_k_up(route_samples)
        new_w_update_up = self.lr * w_grad_up

        update_w_up = tf.assign(self.w, self.w + new_w_update_up)

        w_grad_down = self.CD_k_down(active_samples)
        new_w_update_down = self.lr * w_grad_down

        update_w_down = tf.assign(self.w, self.w + new_w_update_down)

        return [update_w_up, update_w_down]

    '''
    Conditional likelihood of data, good way to track and make sure posterior predictive realizations are close to
    what they should be.
    '''
    def likelihoods(self, route_samples, active_samples):

        #active means to just keep stuff simple
        active_means = tf.reduce_mean(tf.cast(active_samples, tf.float32), axis=0)
        normal = tfd.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma)
        norm_prob = normal.prob(active_means)

        #reshape route samples so that theta can be fit and -1 to account for number of samples, just take mean to keep it simple
        multinomial = tfd.Multinomial(total_count=tf.constant([1.]*self.n_position, dtype=tf.float32), probs=self.theta, validate_args=True)
        probs = multinomial.prob(tf.reshape(tf.cast(route_samples, tf.float32), [-1, self.n_position, self.n_route]))
        multi_prob = tf.reduce_mean(probs)

        return [multi_prob, norm_prob]

    '''
    Free energy computed according to Restricted Boltzmann Machine website
    '''
    def free_energy(self, route_samples, active_samples):

        #use means
        active_means = tf.reshape(tf.reduce_mean(tf.cast(active_samples, tf.float32), axis=0), [1, -1])
        route_means = tf.reshape(tf.reduce_mean(tf.cast(route_samples, tf.float32), axis=0), [-1, 1])

        energy = tf.reshape(tf.negative(tf.matmul(active_means, tf.matmul(self.w, route_means))), [])

        return [energy]



def train(train_data, n_routes, n_positions, n_actives, epochs = 10):
    # directories to save samples and logs
    logs_dir = './logs'

    rbm = RBM(n_route=n_routes, n_position=n_positions, n_active=n_actives)

    # computation graph definition
    x1, x2 = tf.placeholder(tf.float32, shape=[None, n_routes * n_positions]), tf.placeholder(tf.float32, shape=[None, n_actives])
    step = rbm.learn(x1, x2)
    li = rbm.likelihoods(x1, x2)
    fe = rbm.free_energy(x1, x2)

    saver = tf.train.Saver()

    kf = KFold(n_splits=10, shuffle=True)
    routes = train_data['routes']
    actives = train_data['actives']

    # main loop
    with tf.Session() as sess:
        mean_likelihoods = []
        free_energies = []

        init = tf.global_variables_initializer()
        sess.run(init)

        for e in range(epochs):
            # draw samples
            for batch_number, (_, batch_indices) in enumerate(kf.split(routes)):
                batch_route, batch_active = routes[batch_indices], actives[batch_indices]
                sess.run(step, feed_dict={x1: batch_route, x2: batch_active, rbm.lr: 0.1})
                likelihoods = sess.run(li, feed_dict={x1: batch_route, x2: batch_active})
                energy = sess.run(fe, feed_dict={x1: batch_route, x2: batch_active})
                mean_likelihoods.append(likelihoods)
                free_energies.append(energy)
                # save model
                if e is not 0 and batch_number is 0:
                    checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e + 1)
                    print('Saved Model.')
                # print pseudo likelihood
                if e is not 0 and batch_number is 0:
                    print('Epoch {} likelihood for multinomial {} likelihood for actives {}'.format(e, np.mean(mean_likelihoods, axis=0)[0], np.mean(mean_likelihoods, axis=0)[1]))
                    print('Epoch {} free energy {}'.format(e, np.mean(free_energies)))
                    mu = rbm.mu.eval()
                    sigma = rbm.sigma.eval()
                    w = rbm.w.eval()
                    theta = rbm.theta.eval()
                    mean_likelihoods = []
                    free_energies = []

        # draw samples when training finished
        # print('Test')
        # samples = sess.run(sampler, feed_dict={x: noise_x})
        # samples = samples.reshape([train_data.batch_size, 28, 28])
        # save_images(samples, [8, 8], os.path.join(samples_dir, 'test.png'))
        # print('Saved samples.')



if __name__ == "__main__":
    data = Data()

    n_position = 3
    num_samples = 300
    p_vector = (.4, .6)
    active_mu = (10, 15, 20, 25)
    n_active = len(active_mu)
    n_route = len(p_vector)


    simulated_data = data.simulate(num_positions=n_position, num_samples=num_samples, p_vector=p_vector, active_mu=active_mu)

    train(simulated_data, n_route, n_position, n_active)

    print('hi')





