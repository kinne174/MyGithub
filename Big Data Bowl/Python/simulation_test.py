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
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import matplotlib.pyplot as plt
import pandas as pd

tfd = tfp.distributions

class Data:
    '''
    I want this to be able to get the real data and simulate data
    '''
    def __init__(self):
        pass

    def simulate(self, num_routes=3, num_samples=300, p_vector=(.4, .6), active_mu = (10, 15, 20, 25), active_sigma = (2, 1.5, 1.5, 2), success_probability=0.5, seed=635):
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

        num_positions = len(p_vector)
        num_active = len(active_mu)

        #routes is a matrix of dimension num_samples by (num_positions x num_routes)
        routes = np.empty([num_samples, num_positions*num_routes])
        for i in range(num_positions):
            routes[:, list(range(i*num_routes, (i+1)*num_routes))] = np.random.multinomial(1, p_vector, size=num_samples)

        for i in range(num_samples):
            assert sum(routes[i, :]) == num_routes

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
    flag = True #flipping index for computing likelihood

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
        self.lr = tf.placeholder(tf.float32)

        #weight matrix
        self.w = weight([n_active, n_position*n_route], 'W')

        #constant parameters
        self.mu_not = tf.constant([0.]*n_active, dtype=tf.float32)
        self.sigma_not = tf.constant([1.]*n_active, dtype=tf.float32)
        self.eta = tf.constant([1.]*n_active, dtype=tf.float32)
        self.nu = tf.constant([1.]*n_active, dtype=tf.float32)
        self.alpha = tf.constant([1.]*n_route)

        #variable parameters
        self.mu = tf.Variable([0.]*n_active, name='mu')
        self.sigma = tf.Variable([1.]*n_active, name='sigma')
        self.theta = tf.Variable(np.full([n_position, n_route], n_route**-1), name='theta')


    '''
    Posterior predicitve functions. 
    The down generates active realizations based on updated variable parameters (mu and sigma) and initial route realizations.
    The up generates route realizations based on updated variable parameters (theta).
    '''
    def postpreddown(self, routes):

        #number of samples to generate
        num_samples = routes.shape[0]

        #use mean of samples
        routes_mean = tf.reduce_mean(routes, axis=0)

        #location and scale vectors according to predictive posterior calculations
        mean_vec = tf.add(tf.multiply(tf.divide(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.add(tf.square(self.sigma), tf.multiply(num_samples, tf.square(self.sigma_not)))), tf.matmul(self.w, tf.transpose(routes_mean))),
                          tf.multiply(tf.divide(tf.square(self.sigma), tf.add(tf.square(self.sigma), tf.multiply(num_samples, tf.square(self.sigma_not)))), self.mu_not))
        sigma_vec = tf.sqrt(tf.add(tf.divide(tf.multiply(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma)), tf.add(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma))), tf.square(self.sigma)))

        assert mean_vec.shape == self.mu.shape, 'mu shapes for posterior predictive of normal-normal are different'
        assert sigma_vec.shape == self.sigma.shape, 'sigma shapes for posterior predictive of normal-normal are different'

        #create mulivariate normal distribution assuming independence between observations
        multi_normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)

        #should be a tensor with dimension number of samples by number of active data points
        return multi_normal.sample(sample_shape=[num_samples, self.n_active])

    def postpredup(self, num_samples):

        #initialize output
        real_out = tf.zeros(shape=[num_samples, self.n_route*self.n_position])

        for j in range(num_samples):

            #to generate one sample, generate observations for each position route combination
            temp = tf.zeros(shape=self.theta.shape)
            for i in range(self.n_position):
                multinomial = tfd.Multinomial(total_count=1., probs=self.theta[i, :])
                temp[i, :] = multinomial.sample(sample_shape=1)

            #flatten observations to fit into single row
            real_out[j, :] = tf.reshape(temp, [-1])

        return real_out

    '''
    Fix gradients to incorporate the difference between the true observations at the beginning and the new ones at the end.
    Similar to how it was done online. Will need to re write this part. No comments for now.
    '''
    def gradientup(self, route_samples):
        out = tf.zeros(shape=self.w.shape)
        num_samples = route_samples.shape[0]
        for i in range(num_samples):
            new = tf.matmul(tf.matmul(tf.diag(self.sigma),
                                      tf.subtract(tf.matmul(self.w, tf.transpose(route_samples[i, :])), self.mu)),
                            route_samples[i, :])
            out = tf.add(out, new)
        return out

    def gradientdown(self, active_samples):
        out = tf.zeros(shape=self.w.shape)
        num_samples = active_samples.shape[0]
        for i in range(num_samples):
            new = tf.matmul(tf.transpose(active_samples[i, :]), tf.log(tf.divide(tf.reshape(self.theta, [-1]), tf.subtract(1, tf.reshape(self.theta, [-1])))))
            out = tf.add(out, new)
        return out

    '''
    Contrastive Divergence to train the weight matrix in the middle. Goes through k iterations of passing posterior
    predictives back and forth and updating the parameters of the normal-normal model. Not clear how to do this for
    the dirichlet-multinomial distribution. Also might want to look into keeping track of the chain to plot later
    to make sure that everything is mixing well.
    '''
    def CD_k(self, samples):

        assert samples.shape[0] > 0, 'Need atleast one sample'

        #alternate between starting from routes and starting from actives
        if self.flag:
            self.flag = not self.flag

            assert samples.shape[1] == self.n_position*self.n_route, 'Number of observations in the route sample is not correct'

            return self.CD_k_up(samples)
        else:
            self.flag = not self.flag

            assert samples.shape[1] == self.n_active, 'Number of observations in the active sample is not correct'

            return self.CD_k_down(samples)

    def CD_k_up(self, route_samples):

        #given true route samples trade back and forth and produce new route samples at the end
        num_samples = route_samples.shape[0]

        for _ in range(self.k):

            #use the means to compact information
            routes_mean = tf.reduce_mean(route_samples, axis=0)

            for _ in range(self.m):

                #get the mean vec according to posterior of mu
                mean_vec = tf.add(tf.multiply(tf.divide(tf.multiply(num_samples, tf.square(self.sigma_not)),
                                                        tf.add(tf.square(self.sigma),
                                                               tf.multiply(num_samples, tf.square(self.sigma_not)))),
                                              tf.matmul(self.w, tf.transpose(routes_mean))),
                                  tf.multiply(tf.divide(tf.square(self.sigma), tf.add(tf.square(self.sigma),
                                                                                      tf.multiply(num_samples, tf.square(
                                                                                          self.sigma_not)))), self.mu_not))

                #get the sigma vec according to the posterior of mu
                sigma_vec = tf.sqrt(
                    tf.divide(tf.multiply(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma)),
                              tf.add(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma))))

                assert mean_vec.shape == self.mu.shape
                assert sigma_vec.shape == self.sigma.shape

                #intialize and draw new mu
                normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)
                self.mu.assign(normal.sample(sample_shape=self.mu.shape))

                #get alpha and beta according to posterior of sigma
                concentration = tf.add(self.nu, 0.5)
                rate = tf.divide(tf.add(tf.multiply(2, self.nu), tf.square(tf.subtract(tf.matmul(self.w, tf.transpose(routes_mean)), self.mu))), 2)

                assert concentration.shape == ()
                assert rate.shape == self.sigma.shape

                #draw sigma vector indpendently since I'm assuming independence right now
                sig_list = []
                for i in range(self.sigma.shape[0]):
                    inv_gamma = tfd.InverseGamma(concentration=concentration, rate=rate[i])
                    sig_list.append(inv_gamma.sample())
                self.sigma.assign(tf.convert_to_tensor(sig_list, dtype=tf.float32))

            #draw new active samples, within function uses updated mu and sigma
            active_samples = self.postpreddown(route_samples)

            #work with mean reduction of data
            actives_mean = tf.reduce_mean(active_samples, axis=0)

            #initialize alphas of the dirichlet
            new_alphas = tf.matmul(actives_mean, self.w)

            assert new_alphas.shape[0] == 1, 'New alphas should be a single row'
            assert new_alphas.shape[1] == self.n_route*self.n_position, 'New alphas should have a spot for each route/position combination'

            #draw thetas from dirichlet
            new_thetas = []
            for i in range(self.n_position):
                #add prior alphas to new alphas and draw for each position
                dirichlet = tfd.Dirichlet(concentration=tf.add(self.alpha, new_alphas[list(range(i*self.n_route, (i+1)*self.n_route))]))
                new_thetas.append(dirichlet.sample(sample_shape=self.n_route))

            #reshape flattened thetas to fit into n_positions by n_routes shape
            self.theta.assign(tf.reshape(tf.convert_to_tensor(new_thetas), self.theta.shape))

            #use updated theta to sample routes
            route_samples = self.postpredup(num_samples=num_samples)

        ########UPDATE GRADIENT FUNCTION#####################
        gradient = self.gradientup(route_samples)

        return gradient

    def CD_k_down(self, active_samples):

        # given true active samples trade back and forth and produce new active samples at the end
        num_samples = active_samples.shape[0]

        #see comments above, same but in different order
        for _ in range(self.k):
            actives_mean = tf.reduce_mean(active_samples, axis=0)

            new_alphas = tf.matmul(actives_mean, self.w)

            new_thetas = []
            for i in range(self.n_position):
                dirichlet = tfd.Dirichlet(concentration=tf.add(self.alpha, new_alphas[list(range(i*self.n_route, (i+1)*self.n_route))]))
                new_thetas.append(dirichlet.sample(sample_shape=self.n_route))

            self.theta.assign(tf.reshape(tf.convert_to_tensor(new_thetas), self.theta.shape))

            route_samples = self.postpredup(num_samples=num_samples)

            routes_mean = tf.reduce_mean(route_samples, axis=0)

            for _ in range(self.m):

                mean_vec = tf.add(tf.multiply(tf.divide(tf.multiply(num_samples, tf.square(self.sigma_not)),
                                                        tf.add(tf.square(self.sigma),
                                                               tf.multiply(num_samples, tf.square(self.sigma_not)))),
                                              tf.matmul(self.w, tf.transpose(routes_mean))),
                                  tf.multiply(tf.divide(tf.square(self.sigma), tf.add(tf.square(self.sigma),
                                                                                      tf.multiply(num_samples,
                                                                                                  tf.square(
                                                                                                      self.sigma_not)))),
                                              self.mu_not))
                sigma_vec = tf.sqrt(
                    tf.divide(tf.multiply(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma)),
                              tf.add(tf.multiply(num_samples, tf.square(self.sigma_not)), tf.square(self.sigma))))
                normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)

                self.mu.assign(normal.sample(sample_shape=self.mu.shape))

                concentration = tf.add(self.nu, 0.5)
                rate = tf.divide(tf.add(tf.multiply(2, self.nu),
                                        tf.square(tf.subtract(tf.matmul(self.w, tf.transpose(routes_mean)), self.mu))), 2)

                sig_list = []
                for i in range(self.sigma.shape[0]):
                    inv_gamma = tfd.InverseGamma(concentration=concentration, rate=rate[i])
                    sig_list.append(inv_gamma.sample())
                self.sigma.assign(tf.convert_to_tensor(sig_list, dtype=tf.float32))

            active_samples = self.postpreddown(route_samples)

            gradient = self.gradientdown(active_samples)

            return gradient

    def learn(self, samples):
        #get the gradient and update the weight matrix
        w_grad = self.CD_k(samples)
        new_w_update = self.lr * w_grad

        update_w = tf.assign(self.w, self.w + new_w_update)

        return [update_w]

    '''
    Conditional likelihood of data, good way to track and make sure posterior predictive realizations are close to
    what they should be.
    '''
    def likelihoods(self, route_samples, active_samples):

        #route sums this time to do only one likelhood calculation per position
        route_sums = tf.reduce_sum(route_samples, axis=0)
        active_means = tf.reduce_mean(active_samples, axis=0)

        normal = tfd.MultivariateNormalDiag(loc=self.mu, scale_diag=self.sigma)

        likelihood_temp = 0
        for i in range(self.n_position):
            multinomial = tfd.Multinomial(total_count=route_samples.shape[0], probs=self.theta[i, :])
            likelihood_temp += multinomial.prob(route_sums[list(range(i*self.n_route, (i+1)*self.n_route))])

        return [likelihood_temp, normal.prob(active_means)]


def train(train_data, epochs):
    # directories to save samples and logs
    logs_dir = './logs'







