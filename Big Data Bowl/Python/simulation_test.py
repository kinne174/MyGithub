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
from sklearn.linear_model import LinearRegression

tf.enable_eager_execution()

tfd = tfp.distributions

class Data:
    '''
    I want this to be able to get the real data and simulate data
    '''
    def __init__(self):
        pass

    def simulate(self, num_positions=3, num_samples=300, p_vector=(.4, .6), active_mu=(10, 15, 20, 25), active_sigma=(2, 1.5, 1.5, 2), success_probability=0.5, seed=635):
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

def bias(shape, name='biases'):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

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

        #bias vectors
        self.b = bias([n_active], 'b')
        self.c = bias([n_position*n_route], 'c')

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

    def update_w(self, new_w):
        update = tf.assign(self.w, new_w)
        return [update]

    def update_mu(self, new_mu):
        update = tf.assign(self.mu, new_mu)
        return [update]

    def update_sigma(self, new_sigma):
        update = tf.assign(self.sigma, new_sigma)
        return [update]

    def update_theta(self, new_theta):
        update = tf.assign(self.theta, new_theta)
        return [update]

    '''
    Posterior predicitve functions. 
    The down generates active realizations based on updated variable parameters (mu and sigma) and initial route realizations.
    The up generates route realizations based on updated variable parameters (theta).
    '''
    def postpreddown(self, routes):
        #outputs normal samples

        routes = tf.cast(tf.convert_to_tensor(routes), tf.float32)

        #number of samples to generate
        num_samples = tf.cast(tf.shape(routes)[0], tf.float32)

        routes = tf.reshape(routes, [-1, tf.cast(num_samples, tf.int32)])

        # assert routes.get_shape().as_list() == [self.n_position*self.n_route, tf.cast(num_samples, tf.int32)], 'Routes mean should have {} elements but has {}'.format(self.n_position*self.n_route, routes.get_shape().as_list()[1])

        #location and scale vectors according to predictive posterior calculations
        active_mean = tf.reduce_mean(tf.matmul(self.w, routes), axis = 1)
        mean_vec = tf.add(tf.multiply(tf.divide(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.add(tf.square(self.sigma), tf.multiply(num_samples, tf.square(self.sigma_not)))), tf.reshape(active_mean, [-1])),
                          tf.multiply(tf.divide(tf.square(self.sigma), tf.add(tf.square(self.sigma), tf.multiply(num_samples, tf.square(self.sigma_not)))), self.mu_not))

        sigma_vec = tf.sqrt(tf.add(tf.divide(tf.multiply(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma)),
                          tf.add(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma))), tf.square(self.sigma)))

        # assert mean_vec.get_shape().as_list() == self.mu.get_shape().as_list(), 'mu shapes for posterior predictive of normal-normal are different'
        # assert sigma_vec.get_shape().as_list() == self.sigma.get_shape().as_list(), 'sigma shapes for posterior predictive of normal-normal are different'

        #create mulivariate normal distribution assuming independence between observations
        multi_normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)

        #should be a tensor with dimension number of samples by number of active data points
        out = multi_normal.sample(sample_shape=tf.cast(num_samples, tf.int32))
        return out

    def postpredup(self, actives):
        #outputs multinomial samples

        #cast so sample can handle the input
        #number of samples to generate
        num_samples = tf.cast(tf.shape(actives)[0], tf.int32)

        #initialize multinomial distribution drawing one route per position
        multinomial = tfd.Multinomial(total_count=tf.constant([1.]*self.n_position, dtype=tf.float32), probs=self.theta, validate_args=True)

        #draw samples and reshape to conform to structure I've been using
        sample = multinomial.sample(num_samples)
        return tf.reshape(sample, [num_samples, self.n_position*self.n_route])

    '''
    Gradient with respect to the log likelihood of a normal or multinomial distribution depending on direction
    '''
    def gradientup(self, route_samples):
        # in1 = tf.matmul(self.w, tf.transpose(route_samples))
        # in2 = tf.subtract(in1, tf.reshape(self.mu, [-1, 1]))
        # in3 = tf.matmul(in2, route_samples)
        # in4 = tf.diag(tf.reciprocal(self.sigma))
        # in5 = tf.matmul(in4, in3)
        #
        # # out = tf.multiply(-0.5, in5)
        route_samples = tf.cast(tf.convert_to_tensor(route_samples), tf.float32)
        num_samples = tf.shape(route_samples)[0]
        mus = tf.broadcast_to(tf.reshape(self.mu, [-1, 1]), [self.n_active, num_samples])
        out = tf.transpose(tf.linalg.solve(matrix=tf.matmul(tf.transpose(route_samples), route_samples), rhs=tf.matmul(tf.transpose(route_samples), tf.transpose(mus))))

        return out

    def gradientdown(self, active_samples):
        active_samples = tf.cast(tf.convert_to_tensor(active_samples), tf.float32)
        num_samples = tf.gather(tf.shape(active_samples), 0)

        # out = tf.matmul(tf.transpose(active_samples), tf.broadcast_to(tf.log(tf.divide(tf.reshape(self.theta, [1, self.n_position*self.n_route]), tf.subtract(1., tf.reshape(self.theta, [1, self.n_position*self.n_route]))))
        #                                                               , tf.stack([num_samples, tf.convert_to_tensor(self.n_position*self.n_route, dtype=tf.int32)], 0)))
        out = tf.matmul(tf.transpose(active_samples), tf.broadcast_to(self.theta, tf.stack([num_samples, tf.convert_to_tensor(self.n_position*self.n_route, dtype=tf.int32)], 0)))

        assert out.get_shape().as_list() == self.w.get_shape().as_list(), 'shape of gradient update using active samples is not the same as for w'

        return out

    '''
    These functions will compute a gibbs step for the normal-normal and dirichlet-multinomial model. Will be like learn()
    where it will output a variable with tf.assign() call so that sess.run() can update it in for loops.
    '''
    def gibbs_up(self, active_samples):
        active_samples = tf.cast(tf.convert_to_tensor(active_samples), tf.float32)

        num_samples_int = tf.shape(active_samples)[0]
        num_samples_float = tf.cast(num_samples_int, tf.float32)

        active_mean = tf.cast(tf.reshape(tf.reduce_mean(active_samples, axis=0), [-1, 1]), tf.float32)

        # get the mean vec according to posterior of mu
        mean_vec = tf.add(tf.multiply(tf.divide(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.add(tf.square(self.sigma), tf.multiply(num_samples_float, tf.square(self.sigma_not)))), tf.reshape(active_mean, [-1])),
                          tf.multiply(tf.divide(tf.square(self.sigma), tf.add(tf.square(self.sigma), tf.multiply(num_samples_float, tf.square(self.sigma_not)))), self.mu_not))

        # get the sigma vec according to the posterior of mu
        sigma_vec = tf.sqrt(tf.divide(tf.multiply(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.square(self.sigma)),
                                      tf.add(tf.multiply(num_samples_float, tf.square(self.sigma_not)), tf.square(self.sigma))))

        assert mean_vec.get_shape().as_list() == self.mu.get_shape().as_list(), 'Mean vector should have same shape as mu in normal-normal'
        assert sigma_vec.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Scale vector should have same shape as sigma in normal-normal'

        # intialize and draw new mu
        normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)
        mu_update = tf.assign(self.mu, normal.sample())

        # get alpha and beta according to posterior of sigma
        concentration = tf.reshape(tf.reshape(tf.add(self.eta, tf.divide(num_samples_float, 2.)), [1, -1]), [-1])
        in1 = tf.broadcast_to(tf.reshape(self.mu, [1, -1]), tf.shape(active_samples))
        in2 = tf.subtract(active_samples, in1)
        in3 = tf.reduce_sum(tf.square(in2), axis=0)
        rate = tf.reshape(tf.divide(tf.add(tf.reshape(tf.multiply(2., self.nu), [1, -1]), in3), 2.), [-1])

        assert concentration.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Alpha of inverse gamma should have as many elements as sigma'
        assert rate.get_shape().as_list() == self.sigma.get_shape().as_list(), 'Beta of inverse gamma should have as many elements as sigma'

        # draw sigma vector indpendently since I'm assuming independence right now
        inv_gamma = tfd.InverseGamma(concentration=concentration, rate=rate)
        out = inv_gamma.sample()
        sigma_update = tf.assign(self.sigma, tf.sqrt(tf.minimum(out, tf.Variable([np.inf]*self.n_active, dtype=tf.float32))))

        return [mu_update, sigma_update]

    def gibbs_down(self, route_samples):
        route_samples = tf.cast(tf.convert_to_tensor(route_samples), tf.float32)

        # work with mean reduction of data
        route_mean = tf.cast(tf.reshape(tf.reduce_mean(route_samples, axis=0), [1, -1]), tf.float32)

        # initialize alphas of the dirichlet
        new_alphas = route_mean

        assert new_alphas.get_shape().as_list()[0] == 1, 'New alphas should be a single row'
        assert new_alphas.get_shape().as_list()[1] == self.n_route * self.n_position, 'New alphas should have a spot for each route/position combination'

        # draw thetas from dirichlet
        new_thetas = []
        for i in range(self.n_position):
            # add prior alphas to new alphas and draw for each position
            dirichlet = tfd.Dirichlet(concentration=tf.add(self.alpha, new_alphas[0, i * self.n_route:(i + 1) * self.n_route]))
            new_thetas.append(dirichlet.sample())

        # reshape flattened thetas to fit into n_positions by n_routes shape
        theta_update = tf.assign(self.theta, tf.divide(self.theta + tf.reshape(tf.convert_to_tensor(new_thetas), tf.shape(self.theta)), 2.))

        return [theta_update]

    '''
    CHANGE THESE TO ONLY UPDATE TO INCREASE TOWARDS TRUE LIKELIHOOD
    '''
    def learnup(self, true_samples, generated_samples):
        #route samples
        true_samples = tf.cast(tf.convert_to_tensor(true_samples), tf.float32)
        num_samples = tf.cast(tf.gather(tf.shape(true_samples), 0), tf.float32)

        gradient = tf.subtract(self.gradientup(true_samples), self.gradientup(generated_samples))

        assert gradient.get_shape().as_list() == self.w.get_shape().as_list(), 'Gradient should have the same shape as weight matrix'

        return gradient

    def learndown(self, true_samples, generated_samples):
        #active samples
        true_samples = tf.cast(tf.convert_to_tensor(true_samples), tf.float32)
        num_samples = tf.cast(tf.gather(tf.shape(true_samples), 0), tf.float32)

        gradient = tf.subtract(self.gradientdown(true_samples), self.gradientdown(generated_samples))

        assert gradient.get_shape().as_list() == self.w.get_shape().as_list(), 'Gradient should have the same shape as weight matrix'

        new_w_update = self.lr * gradient

        return new_w_update

    '''
    Conditional likelihood of data, good way to track and make sure posterior predictive realizations are close to
    what they should be.
    '''
    def likelihoods(self, route_samples, active_samples):

        #active means to just keep stuff simple
        active_samples = tf.cast(active_samples, tf.float32)
        normal = tfd.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma)
        norm_prob = normal.prob(active_samples)
        norm_prob = tf.reduce_mean(norm_prob)

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


def warm_start_w(y_active, X_active, y_route, X_route):
    active_on_route = LinearRegression().fit(X_route, y_active)
    route_on_active = LinearRegression().fit(X_active, y_route)

    out_w = (active_on_route.coef_ + np.transpose(route_on_active.coef_))/2.
    out_b = active_on_route.intercept_
    out_c = route_on_active.intercept_

    assert out_w.shape == (y_active.shape[1], y_route.shape[1]), 'warm start weight matrix is not the correct dimension'

    return [out_w, out_b, out_c]

def train(train_data, n_routes, n_positions, n_actives, m = 50, k = 1, epochs = 10):
    # directories to save samples and logs
    logs_dir = './logs'

    rbm = RBM(n_route=n_routes, n_position=n_positions, n_active=n_actives)

    # computation graph definition
    x1, x2 = tf.placeholder(tf.float32, shape=[None, n_routes * n_positions]), tf.placeholder(tf.float32, shape=[None, n_actives])
    x11, x22 = tf.placeholder(tf.float32, shape=[None, n_routes * n_positions]), tf.placeholder(tf.float32, shape=[None, n_actives])
    li = rbm.likelihoods(x1, x2)
    fe = rbm.free_energy(x1, x2)
    gu = rbm.gibbs_up(x2)
    gd = rbm.gibbs_down(x1)
    ppu = rbm.postpredup(x2)
    ppd = rbm.postpreddown(x1)
    lu = rbm.learnup(x1, x11)
    ld = rbm.learndown(x2, x22)

    saver = tf.train.Saver()

    kf = KFold(n_splits=10, shuffle=True)
    routes = train_data['routes']
    actives = train_data['actives']

    # main loop
    with tf.Session() as sess:
        mean_likelihoods = []
        # free_energies = []

        init = tf.global_variables_initializer()
        sess.run(init)

        for e in range(epochs):

            # draw samples
            for batch_number, (_, batch_indices) in enumerate(kf.split(routes)):
                batch_route, true_route = routes[batch_indices].astype(np.float32), routes[batch_indices].astype(np.float32)
                batch_active = actives[batch_indices].astype(np.float32)

                assert k > 0, 'k must be greater than zero but is {}'.format(k)
                assert m > 0, 'm must be greater than zero but is {}'.format(m)

                for _ in range(m):
                    sess.run(gu, feed_dict={x2: batch_active})

                for _ in range(k):
                    batch_active = sess.run(ppd, feed_dict={x1: batch_route})
                    batch_route = sess.run(ppu, feed_dict={x2: batch_active})

                sess.run(lu, feed_dict={x11: true_route, x1: batch_route, rbm.lr: 0.01})

                batch_route = routes[batch_indices].astype(np.float32)
                batch_active, true_active = actives[batch_indices].astype(np.float32), actives[batch_indices].astype(np.float32)

                sess.run(gd, feed_dict={x1: batch_route})

                for _ in range(k):
                    batch_route = sess.run(ppu, feed_dict={x2: batch_active})
                    batch_active = sess.run(ppd, feed_dict={x1: batch_route})

                sess.run(ld, feed_dict={x22: true_active, x2: batch_active, rbm.lr: 0.01})

                likelihoods = sess.run(li, feed_dict={x1: batch_route, x2: batch_active})
                # energy = sess.run(fe, feed_dict={x1: batch_route, x2: batch_active})
                mean_likelihoods.append(likelihoods)
                # free_energies.append(energy)

                mu = rbm.mu.eval()
                sigma = rbm.sigma.eval()
                w = rbm.w.eval()
                theta = rbm.theta.eval()

                # save model
                if e is not 0 and batch_number is 0:
                    checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e + 1)
                    print('Saved Model.')
                # print pseudo likelihood
                if e is not 0 and batch_number is 0:
                    print('Epoch {} likelihood for multinomial {} likelihood for actives {}'.format(e, np.mean(mean_likelihoods, axis=0)[0], np.mean(mean_likelihoods, axis=0)[1]))
                    # print('Epoch {} free energy {}'.format(e, np.mean(free_energies)))
                    mean_likelihoods = []
                    # free_energies = []

'''
THING TO DO TOMORROW IS TO REWRITE GRADIENT FUNCTION FOR ACTIVE SAMPLES TO REFLECT SOFTMAX APPROACH
ALSO COME UP WITH A WAY TO SAVE UPDATES TO VARIABLES AND w IN A MORE GIBBS SAMPLY WAY
ALSO THE ROUTINE WILL BE TO WARM START w WITH ONE STEP TRUE SAMPLES AND UPDATE, THEN CAN DO MORE OF A GIBBS AND UPDATE (MONITOR THIS TO START THO)
'''



if __name__ == "__main__":
    data = Data()

    n_position = 3
    num_samples = 300
    p_vector = (.2, .6, .2)
    active_mu = (10, 15, 20, 25)
    n_active = len(active_mu)
    n_route = len(p_vector)


    simulated_data = data.simulate(num_positions=n_position, num_samples=num_samples, p_vector=p_vector, active_mu=active_mu)

    rbm = RBM(n_route, n_position, n_active)

    # train(simulated_data, n_route, n_position, n_active)

    actives = simulated_data['actives']
    active_sample = actives[np.random.randint(actives.shape[0],size=500),:].astype(np.float32)

    routes = simulated_data['routes']
    route_sample = routes[np.random.randint(routes.shape[0],size=500),:].astype(np.float32)

    for _ in range(100):
        _ = rbm.gibbs_up(active_sample)

    update_pos = rbm.gradientup(route_sample)

    update_w = update_pos

    u = rbm.update_w(update_w)

    batch_active = rbm.postpreddown(route_sample)

    print('hi')





