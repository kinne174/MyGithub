import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os, glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime

# tf.enable_eager_execution()

tfd = tfp.distributions

class Data:
    '''
    I want this to be able to get the real data and simulate data
    '''
    def __init__(self, seed=635):
        #set seed
        np.random.seed(seed)

    def simulate(self, n_position=3, n_active = 4, n_sample=300, p_vector=(.4, .6)):
        '''
        :param n_position: number of multinomail RVs to draw
        :param n_active: number of active RVs
        :param n_sample: number of samples
        :param p_vector: probability vector for mulitinomials, tuple
        :param success_probability: probabilty of route success
        :param seed: random seed
        :return: dictionary of route samples and active samples and success responses and true beta
        '''

        assert  round(sum(p_vector), 4) == 1, "Sum of probability vector not equal to one: {}".format(sum(p_vector))

        n_route = len(p_vector)

        #routes is a matrix of dimension num_samples by (num_positions x num_routes)
        routes = np.empty([n_sample, n_position*n_route])
        for i in range(n_position):
            routes[:, list(range(i*n_route, (i+1)*n_route))] = np.random.multinomial(1, p_vector, size=n_sample)

        for i in range(n_sample):
            assert sum(routes[i, :]) == n_position, 'The num routes is {} and the sum is {}'.format(n_route, sum(routes[i, :]))

        true_beta1 = np.random.uniform(low=-.5, high=.5, size=[n_active, n_position*n_route])
        true_beta2 = np.arange(1, n_active + 1).reshape([1, n_active])
        bias = np.random.uniform(low=-1, high=1, size=[1, 1])

        actives = np.add(np.transpose(np.matmul(true_beta1, np.transpose(routes))), np.random.normal(scale=0.2, size=[n_sample, n_active]))

        actives = scale(actives)

        success_probability = expit(np.add(np.matmul(true_beta2, np.transpose(actives)), bias)).reshape(n_sample)

        successes = np.random.binomial(1, p=success_probability, size=None).reshape([-1, 1])

        print('Distribution of successes is {}'.format(np.mean(successes)))

        return {'actives': actives, 'routes': routes, 'sucesses': successes, 'true_beta': true_beta1}

    def real(self):
        header = r'C:/Users/Mitch/Documents/UofM/Fall 2018/NFL/Data/all_plays'
        os.chdir(header)
        all_dataframes = []
        for f in glob.glob('*.csv'):
            all_dataframes.append(pd.read_csv(f))

        all_plays = pd.concat(all_dataframes)

        named_route_column_names = ['playerL1_route', 'playerL2_route', 'playerL3_route', 'playerM1_route', 'playerM2_route', 'playerR3_route', 'playerR2_route', 'playerR1_route']

        routes = all_plays[named_route_column_names]

        routes_array = self.route_names2one_hots(self.collapse_routes(routes))

        all_active_column_names = ["blitz", "cDef1_outcome_distance", "cDef1_outcome_position", "cDef1_outcome_speed", "cDef1_pass_distance",
                               "cDef1_pass_position", "cDef1_pass_speed", "cDef2_outcome_distance", "cDef2_outcome_position",
                               "cDef2_outcome_speed", "cDef2_pass_distance", "cDef2_pass_position", "cDef2_pass_speed", "closest_sideline",
                               "in_pocket", "intended_direction", "intended_distance", "intended_speed", "num_DBs", "num_closest_defenders_to_qb",
                               "playerL1_depth", "playerL1_position", "playerL2_depth", "playerL2_position",
                               "playerL3_depth", "playerL3_position", "playerM1_depth", "playerM1_position",
                               "playerM2_depth", "playerM2_position", "playerR1_depth", "playerR1_position",
                               "playerR2_depth", "playerR2_position", "playerR3_depth", "playerR3_position",
                               "shotgun", "pass_success", "time_between_pass_and_outcome", "time_between_snap_and_pass"]

        '''could collapse the L1, L2,... data'''

        active_column_names = ["cDef1_outcome_distance", "cDef1_outcome_speed", "cDef1_pass_distance", "cDef1_pass_speed",
                               "cDef2_outcome_distance", "cDef2_outcome_speed", "cDef2_pass_distance", "cDef2_pass_speed",
                               "closest_sideline", "intended_direction", "intended_distance", "intended_speed",
                               "num_closest_defenders_to_qb", "time_between_pass_and_outcome", "time_between_snap_and_pass"]

        actives = all_plays[active_column_names]

        na_rows = self.na_rows(actives)

        routes_array = np.delete(routes_array, na_rows, axis=0)

        actives = actives.dropna()

        actives_array = actives.to_numpy(dtype=np.float32)

        successes = all_plays['success'].astype(np.int32).values

        successes_out = np.delete(successes, na_rows).reshape([-1, 1])

        scaler = preprocessing.StandardScaler()
        actives_array = scaler.fit_transform(actives_array)

        return {'actives': actives_array, 'routes': routes_array, 'sucesses': successes_out}

    def block2bubble(self, df):
        # want to change 'block.bubble' to 'bubble.block'
        # unneeded apparently...

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                if df.iloc[i, j] == 'block.bubble':
                    df.iloc[i, j] = 'bubble.block'

        return df

    def collapse_routes(self, df):
        # remove NaNs and move route names all the way to the right
        # L2, L1, M, R1, R2 = [], [], [], [], []
        series_list = []

        for i in range(df.shape[0]):
            l = df.iloc[i].dropna().tolist()
            if len(l) > 5 and 'bubble.block' in l:
                l.remove('bubble.block')
            l = l[:5]
            assert len(l) == 5, 'length of list is not correct: {}'.format(len(l))
            series_list.append(l)

        out_df = pd.DataFrame(series_list)

        return out_df

    def route_names2one_hots(self, df):
        # take collapsed df and return a numpy array with 1 in appropriate spot for route

        unique_set = set()

        for j in range(df.shape[1]):
            unique_set = unique_set.union(set(df.iloc[:, j].unique()))

        unique_list = list(unique_set)

        out_array = np.zeros([df.shape[0], len(unique_list)*df.shape[1]])

        for i in range(df.shape[0]):
            ser = df.iloc[i]
            for ii, r in ser.iteritems():
                out_array[i, ii*len(unique_list) + unique_list.index(r)] = 1

        return out_array

    def na_rows(self, df):
        na_df = df.isna()
        na_list = []

        for i in range(na_df.shape[0]):
            if np.any(na_df.iloc[i]):
                na_list.append(i)

        return na_list


def weight(shape, name='weights'):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def bias(shape, name='biases'):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

class Gibbs_RBM:
    def __init__(self, n_route, n_position, n_active, n_sample, n_classes=2, n_hidden=128):
        '''
        :param n_route: Number of routes recorded for each position
        :param n_position: Number of spots the routes could be run from (in my data this is five)
        :param n_active: Number of active datapoints (things like speed, angle, distance, etc.)
        '''

        self.n_route = n_route
        self.n_position = n_position
        self.n_active = n_active
        self.n_sample = n_sample


        # learning rate
        self.lr = tf.placeholder(tf.float32) if not tf.executing_eagerly() else tf.constant(1., dtype=tf.float32)

        # weight matrix
        self.w1 = weight([n_active, n_position*n_route], 'W1')
        self.w2 = weight([n_hidden, n_active], 'W2')
        self.w3 = weight([1, n_hidden], 'W3')
        # self.w4 = weight([n_hidden, n_hidden*2], 'W4')
        # self.w5 = weight([1, n_hidden], 'W5')

        #bias
        '''possibly include bias at first level too??'''
        self.b1 = bias([n_active])
        self.b2 = bias([n_hidden])
        self.b3 = bias([1])
        # self.b4 = bias([n_hidden])
        # self.b5 = bias([1])

        # constant parameters
        self.eta = tf.constant(1., dtype=tf.float32)
        self.nu = tf.constant(1., dtype=tf.float32)
        self.Vinv = tf.constant(np.tril(np.full(shape=[n_active, n_active], fill_value=0.5)) +
                                np.triu(np.full(shape=[n_active, n_active], fill_value=0.5))
                                , dtype=tf.float32)
        self.V = tf.linalg.inv(self.Vinv)

        # variable parameters
        self.beta_routes = tf.Variable(tf.truncated_normal([n_active, n_sample], stddev=0.1), name='beta_routes', dtype=tf.float32)
        self.beta_I = tf.Variable(tf.truncated_normal(self.w1.shape, stddev=0.1), name='beta_I', dtype=tf.float32)
        self.beta_I_not = tf.Variable(tf.truncated_normal(self.w1.shape, stddev=0.1), name='beta_I', dtype=tf.float32)
        self.sigma_routes = tf.Variable(tf.truncated_normal([], mean=1, stddev=0.1), name='sigma_routes', dtype=tf.float32)
        self.sigma_I = tf.Variable(tf.truncated_normal([], mean=1, stddev=0.1), name='sigma_I', dtype=tf.float32)
        self.sigma_I_not = tf.Variable(tf.truncated_normal([], mean=1, stddev=0.1), name='sigma_I', dtype=tf.float32)

        #lists to keep track of variable parameters in a gibbsy way
        self.beta_routes_list = [self.beta_routes, self.beta_routes]
        self.beta_I_list = [self.beta_I, self.beta_I]
        self.sigma_routes_list = [self.sigma_routes]
        self.sigma_I_list = [self.sigma_I]


    def multilayer_perceptron(self, route_sample, bn=True):
        if bn:
            layer_1_ = tf.matmul(route_sample, tf.transpose(self.w1))
            m1, v1 = tf.nn.moments(layer_1_, axes=[0])
            layer_1_batch_norm = tf.nn.batch_normalization(layer_1_, mean=m1, variance=v1, offset=self.b1, variance_epsilon=1e-4, scale=1)
            layer_1 = tf.nn.relu(tf.nn.dropout(layer_1_batch_norm, keep_prob=0.75))

            layer_2_ = tf.matmul(layer_1, tf.transpose(self.w2))
            m2, v2 = tf.nn.moments(layer_2_, axes=[0])
            layer_2_batch_norm = tf.nn.batch_normalization(layer_2_, mean=m2, variance=v2, offset=self.b2, variance_epsilon=1e-4, scale=1)
            layer_2 = tf.nn.relu(tf.nn.dropout(layer_2_batch_norm, keep_prob=0.75))

            # layer_3_ = tf.matmul(layer_2, tf.transpose(self.w3))
            # m3, v3 = tf.nn.moments(layer_3_, axes=[0])
            # layer_3_batch_norm = tf.nn.batch_normalization(layer_3_, mean=m3, variance=v3, offset=self.b3, variance_epsilon=1e-4, scale=1)
            # layer_3 = tf.nn.relu(tf.nn.dropout(layer_3_batch_norm, 0.75))
            #
            # layer_4_ = tf.matmul(layer_3, tf.transpose(self.w4))
            # m4, v4 = tf.nn.moments(layer_4_, axes=[0])
            # layer_4_batch_norm = tf.nn.batch_normalization(layer_4_, mean=m4, variance=v4, offset=self.b4, variance_epsilon=1e-4, scale=1)
            # layer_4 = tf.nn.relu(tf.nn.dropout(layer_4_batch_norm, 0.75))

            out_layer = tf.add(tf.matmul(layer_2, tf.transpose(self.w3)), self.b3)

        else:
            layer_1 = tf.matmul(route_sample, tf.transpose(self.w1))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, tf.transpose(self.w2)), self.b2))
            # layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, tf.transpose(self.w3)), self.b3))
            # layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, tf.transpose(self.w4)), self.b4))
            out_layer = tf.add(tf.matmul(layer_2, tf.transpose(self.w3)), self.b3)

        return out_layer

    def multilayer_perceptron_active(self, route_sample, bn=False):
        if bn:
            layer_1_ = tf.matmul(route_sample, tf.transpose(self.w1))
            m1, v1 = tf.nn.moments(layer_1_, axes=[0])
            layer_1_batch_norm = tf.nn.batch_normalization(layer_1_, mean=m1, variance=v1, offset=None, variance_epsilon=1e-4, scale=1)
            out_layer = layer_1_batch_norm

        else:
            out_layer = tf.matmul(route_sample, tf.transpose(self.w1))

        return out_layer


    def gibbs_beta_routes_mean(self, route_sample):

        assert route_sample.shape[1] == self.n_position*self.n_route, 'route sample column shape is incorrect {}'.format(route_sample.shape[1])
        # assert route_sample.shape[0] == 1, 'route sample row shape is incorrect'

        route_sample = tf.cast(tf.convert_to_tensor(route_sample), tf.float32)

        mean_vec = tf.transpose(tf.matmul(self.w1, tf.transpose(route_sample)))
        sigma_mat = tf.multiply(tf.square(self.sigma_routes), self.Vinv)

        normal = tfd.MultivariateNormalFullCovariance(loc=mean_vec, covariance_matrix=sigma_mat)
        sample = tf.transpose(normal.sample())
        beta_routes_update = tf.assign(self.beta_routes, sample)

        self.beta_routes_list.append(sample)

        return [beta_routes_update]


    def gibbs_beta_routes_variance(self, route_sample):

        assert route_sample.shape[1] == self.n_position*self.n_route, 'route sample column shape is incorrect {}'.format(route_sample.shape[1])
        # assert route_sample.shape[0] == 1, 'route sample row shape is incorrect'

        route_sample = tf.cast(tf.convert_to_tensor(route_sample), tf.float32)

        concentration = tf.add(self.nu, tf.multiply(0.5, tf.cast(self.n_sample * self.n_active, tf.float32)))
        in1 = tf.matmul(tf.subtract(tf.matmul(self.w1, tf.transpose(route_sample)), self.beta_routes),
                                          tf.transpose(tf.subtract(tf.matmul(self.w1, tf.transpose(route_sample)), self.beta_routes)))
        rate = tf.divide(tf.add(tf.multiply(2., self.eta), tf.reduce_sum(tf.linalg.diag_part(
                                in1))), 2.)

        assert rate.get_shape().as_list() == [], 'rate shape incorrect, currently is {}'.format(rate.get_shape().as_list())

        inverse_gamma = tfd.InverseGamma(rate=rate, concentration=concentration)
        sample = tf.sqrt(inverse_gamma.sample())
        sigma_routes_update = tf.assign(self.sigma_routes, sample)

        self.sigma_routes_list.append(sample)

        return [sigma_routes_update]


    def gibbs_beta_mean(self):

        mean_vec = tf.divide(tf.add(tf.multiply(self.sigma_I_not, tf.reshape(self.w1, [-1])), tf.multiply(self.sigma_I, tf.reshape(self.beta_I_not, [-1]))),
                             tf.add(self.sigma_I, self.sigma_I_not))
        sigma_vec = tf.reshape(tf.fill([1, self.n_active*self.n_position*self.n_route],
                                       tf.divide(tf.multiply(self.sigma_I, self.sigma_I_not), tf.add(self.sigma_I, self.sigma_I_not))), [-1])

        normal = tfd.MultivariateNormalDiag(loc=mean_vec, scale_diag=sigma_vec)
        sample = normal.sample()
        beta_update = tf.assign(self.beta_I, tf.reshape(sample, tf.shape(self.w1)))

        self.beta_I_list.append(tf.reshape(sample, tf.shape(self.w1)))

        return [beta_update]


    def gibbs_beta_variance(self):
        concentration = tf.add(self.nu, tf.multiply(0.5, self.n_active * self.n_position * self.n_route))
        rate = tf.divide(tf.add(tf.multiply(2., self.eta),
                                tf.matmul(tf.subtract(tf.reshape(self.w1, [1, -1]), tf.reshape(self.beta_I, [1, -1])),
                                          tf.transpose(tf.subtract(tf.reshape(self.w1, [1, -1]), tf.reshape(self.beta_I, [1, -1]))))),
                         2.)

        assert rate.get_shape().as_list() == [1,1], 'rate shape incorrect, currently is {}'.format(rate.get_shape().as_list())

        inverse_gamma = tfd.InverseGamma(rate=tf.reshape(rate, []), concentration=concentration)
        sample = tf.sqrt(inverse_gamma.sample())
        sigma_I_update = tf.assign(self.sigma_I, sample)

        self.sigma_I_list.append(sample)

        return [sigma_I_update]


    def gibbs_multinomial(self, route_sample):
        assert route_sample.shape[1] == self.n_position * self.n_route, 'route sample column shape is incorrect {}'.format(route_sample.shape[1])

        route_sample = tf.cast(tf.convert_to_tensor(route_sample), tf.float32)

        route_sums = tf.reduce_mean(route_sample, axis=0)

        route_sums = tf.reshape(route_sums, [self.n_position, self.n_route])

        dirichlet_multinomial = tfd.DirichletMultinomial(total_count=1, concentration=route_sums)

        sample = dirichlet_multinomial.sample(sample_shape=self.n_sample)

        sample = tf.reshape(sample, [-1, self.n_position*self.n_route])

        assert sample.get_shape().as_list()[1] == self.n_position*self.n_route, 'shape of sample is incorrect: {}'.format(sample.get_shape().to_list)
        # assert tf.reduce_sum(sample) == tf.cast(self.n_position, tf.float32), 'there are not the correct number of elements in the sample: {}'.format(tf.reduce_sum(sample))

        return sample


    def learn(self, lam1, lam2, lam3, route_samples, active_samples):

        # if tf.executing_eagerly():
        #     assert 0 <= lam1 <= 1, 'lam1 needs to be between 0 and 1 and is {}'.format(lam1)
        #     assert 0 <= lam2 <= 1, 'lam2 needs to be between 0 and 1 and is {}'.format(lam2)
        #     assert 0 <= lam3 <= 1, 'lam3 needs to be between 0 and 1 and is {}'.format(lam3)
        #     assert lam1 + lam2 + lam3 == 1, 'lambdas need to add up to 1'

        assert route_samples.shape[1] == self.n_position*self.n_route, 'route samples shape is incorrect {}'.format(route_samples.shape[1])
        assert active_samples.shape[1] == self.n_active, 'active samples shape is incorrect {}'.format(active_samples.shape[1])

        lam1 = tf.cast(lam1, tf.float32)
        lam2 = tf.cast(lam2, tf.float32)
        lam3 = tf.cast(lam3, tf.float32)

        route_samples = tf.cast(tf.convert_to_tensor(route_samples), tf.float32)
        active_samples = tf.cast(tf.convert_to_tensor(active_samples), tf.float32)

        '''
        PUT MEAN OF BETAS HERE
        '''
        beta_routes_mean = tf.reshape(tf.reduce_mean(tf.convert_to_tensor(self.beta_routes_list), axis=0), tf.shape(self.beta_routes))
        beta_I_mean = tf.reshape(tf.reduce_mean(tf.convert_to_tensor(self.beta_I_list), axis=0), tf.shape(self.beta_I))
        sigma_routes_mean = tf.reduce_mean(tf.convert_to_tensor(self.sigma_routes_list))
        sigma_I_mean = tf.reduce_mean(tf.convert_to_tensor(self.sigma_I_list))

        mat = tf.add(tf.multiply(tf.multiply(lam1, tf.reciprocal(tf.square(sigma_routes_mean))), self.V),
                     tf.add(tf.multiply(tf.multiply(lam2, tf.reciprocal(tf.square(sigma_I_mean))), tf.eye(self.n_active)),
                            tf.multiply(lam3, tf.eye(self.n_active)))
                     )

        rhs1 = tf.add(tf.multiply(tf.multiply(lam1, tf.reciprocal(tf.square(sigma_routes_mean))), tf.matmul(self.V, tf.matmul(beta_routes_mean, route_samples))),
                      tf.add(tf.multiply(tf.multiply(lam2, tf.reciprocal(tf.square(sigma_I_mean))), beta_I_mean),
                             tf.multiply(lam3, tf.matmul(tf.transpose(active_samples), route_samples)))
                      )
        rhs2 = tf.linalg.inv(tf.add(tf.multiply(lam1, tf.matmul(tf.transpose(route_samples), route_samples)),
                                    tf.add(tf.multiply(lam2, tf.eye(self.n_position*self.n_route)),
                                           tf.multiply(lam3, tf.matmul(tf.transpose(route_samples), route_samples)))
                                    )
                             )
        rhs = tf.matmul(rhs1, rhs2)

        gradient = tf.linalg.solve(matrix=mat, rhs=rhs)

        w_update = tf.assign(self.w1, gradient)

        beta_I_not_update = tf.assign(self.beta_I_not, beta_I_mean)
        sigma_I_not_update = tf.assign(self.sigma_I_not, sigma_I_mean)

        return [w_update, beta_I_not_update, sigma_I_not_update]


    def likelihood(self, route_sample):
        assert route_sample.shape[1] == self.n_position * self.n_route, 'route sample column shape is incorrect {}'.format(route_sample.shape[1])

        beta_routes_mean = tf.reduce_mean(tf.convert_to_tensor(self.beta_routes_list), axis=0)
        beta_I_mean = tf.reduce_mean(tf.convert_to_tensor(self.beta_I_list), axis=0)
        sigma_routes_mean = tf.reduce_mean(tf.convert_to_tensor(self.sigma_routes_list))
        sigma_I_mean = tf.reduce_mean(tf.convert_to_tensor(self.sigma_I_list))

        mean_vec1 = tf.transpose(beta_routes_mean)
        sigma_mat = tf.multiply(sigma_routes_mean, self.Vinv)
        normal1 = tfd.MultivariateNormalFullCovariance(loc=mean_vec1, covariance_matrix=sigma_mat)

        prob1 = normal1.prob(tf.transpose(tf.matmul(self.w1, tf.transpose(route_sample))))
        prob1 = tf.reduce_mean(prob1)

        mean_vec2 = tf.reshape(beta_I_mean, [-1])
        sigma_vec = tf.reshape(tf.fill([1, self.n_active*self.n_position*self.n_route], sigma_I_mean), [-1])
        normal2 = tfd.MultivariateNormalDiag(loc=mean_vec2, scale_diag=sigma_vec)

        prob2 = normal2.prob(tf.reshape(self.w1, [-1]))

        return [prob1, prob2]

    def reset_lists(self, partial):
        if not partial:
            self.beta_I_list = []
            self.sigma_I_list = []

        self.beta_routes_list = []
        self.sigma_routes_list = []



def train(train_data, n_routes, n_positions, n_actives, n_sample, k=1, epochs=10):
    # directories to save samples and logs
    os.chdir(r'C:\Users\Mitch\PycharmProjects\NFL')
    logs_dir = './logs'
    assert os.path.exists(logs_dir)

    # true beta
    true_beta = train_data['true_beta'].astype(np.float32) if 'true_beta' in train_data else None

    n_sample = np.int32(np.ceil(n_sample))
    gibbs_rbm = Gibbs_RBM(n_route=n_routes, n_position=n_positions, n_active=n_actives, n_sample=n_sample)

    x1, x2 = tf.placeholder(tf.float32, shape=[None, n_routes * n_positions]), tf.placeholder(tf.float32, shape=[None, n_actives])
    l1, l2, l3 = tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=()), tf.placeholder(tf.float32, shape=())

    step = gibbs_rbm.learn(l1, l2, l3, x1, x2)
    gbrm = gibbs_rbm.gibbs_beta_routes_mean(x1)
    gbrv = gibbs_rbm.gibbs_beta_routes_variance(x1)
    gbm = gibbs_rbm.gibbs_beta_mean()
    gbv = gibbs_rbm.gibbs_beta_variance()
    gm = gibbs_rbm.gibbs_multinomial(x1)
    li = gibbs_rbm.likelihood(x1)

    saver = tf.train.Saver()

    routes = train_data['routes']
    actives = train_data['actives']
    sucesses = train_data['sucesses']

    train_routes, test_routes, train_actives, test_actives, train_successes, test_successes = train_test_split(routes, actives, sucesses, test_size=0.2, random_state=635)

    out2 = -1.
    # lam1_start, lam2_start, lam3_start = 0.0, 0.2, 0.8
    # lam1_end, lam2_end, lam3_end = 0.6, 0.4, 0.0
    # lam1_delta, lam2_delta, lam3_delta = (lam1_end - lam1_start)/epochs, (lam2_end - lam2_start)/epochs, (lam3_end - lam3_start)/epochs
    # lam1, lam2, lam3 = lam1_start, lam2_start, lam3_start

    # MLP
    y = tf.placeholder(tf.float32, shape=[None, 1])
    logits = gibbs_rbm.multilayer_perceptron(x1)
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op)

    pred_active = gibbs_rbm.multilayer_perceptron_active(x1)
    loss_active = tf.losses.mean_squared_error(labels=x2, predictions=pred_active)
    optimizer_active = tf.train.AdamOptimizer(learning_rate=0.3)
    descent = optimizer_active.minimize(loss_active)

    with tf.Session() as sess:
        mean_likelihoods = []
        costs_success = []
        costs_actives = []

        init = tf.global_variables_initializer()
        sess.run(init)

        for e in range(epochs):

            for batch_number in range(10):
                batch_indices = np.random.choice(train_routes.shape[0], size=n_sample)
                batch_active, batch_route, batch_success = train_actives[batch_indices].astype(np.float32), train_routes[batch_indices].astype(np.float32), train_successes[batch_indices].astype(np.float32)

                assert k > 0, 'k must be greater than zero but is {}'.format(k)

                # supervised

                for _ in range(k):
                    sess.run(gbrm, feed_dict={x1: batch_route})
                    sess.run(gbrv, feed_dict={x1: batch_route})
                    sess.run(gbm)
                    sess.run(gbv)

                sess.run(step, feed_dict={l1: 0.2, l2: 0.2, l3: 0.6, x1: batch_route, x2: batch_active})

                # beta_routes = gibbs_rbm.beta_routes.eval()
                # beta_I = gibbs_rbm.beta_I.eval()
                # sigma_routes = gibbs_rbm.sigma_routes.eval()
                # sigma_I = gibbs_rbm.sigma_I.eval()
                # w1 = gibbs_rbm.w1.eval()

                _, cs = sess.run([train_op, loss_op], feed_dict={x1: batch_route, y: batch_success})

                costs_success.append(cs)

                # _, ca = sess.run([descent, loss_active], feed_dict={x1: batch_route, x2: batch_active})
                #
                # costs_actives.append(ca)

                gibbs_rbm.reset_lists(partial=True)

                # unsupervised
                batch_route_us = sess.run(gm, feed_dict={x1: batch_route})

                for _ in range(k):
                    sess.run(gbrm, feed_dict={x1: batch_route_us})
                    sess.run(gbrv, feed_dict={x1: batch_route_us})
                    sess.run(gbm)
                    sess.run(gbv)

                sess.run(step, feed_dict={l1: 1.4, l2: 1.6, l3: 0.0, x1: batch_route_us, x2: batch_active})

                # mean_likelihoods.append(sess.run(li, feed_dict={x1: batch_route}))

                gibbs_rbm.reset_lists(partial=False)

                # #update lambdas
                # lam1 += lam1_delta
                # lam2 += lam2_delta
                # lam3 += lam3_delta

                if 'true_beta' in train_data:
                    out2 = tf.reduce_mean(tf.square(tf.subtract(gibbs_rbm.w1.eval(), tf.convert_to_tensor(true_beta)))).eval()
                    # print('hi')

                if e is not 0 and (e + 1) % 20 == 0 and batch_number is 0 and False:
                    checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e + 1)
                    print('Saved Model.')

                if e is not 0 and (e + 1) % 50 == 0 and batch_number is 0:
                    in1 = tf.matmul(gibbs_rbm.w1.eval(), tf.transpose(batch_route)).eval()
                    in2 = tf.transpose(batch_active).eval()
                    out3 = tf.reduce_mean(tf.abs(
                        tf.subtract(in1, in2))).eval()
                    print('Epoch {} cost actives is {:.3f}, cost success is {:.3f}, closeness to actives is {:.4f}'.format(e, np.mean(costs_actives), np.mean(costs_success), out3))
                    # mean_likelihoods = []
                    costs_actives = [1.]
                    costs_success = []

                if e is not 0 and (e + 1) % 100 == 0 and batch_number is 0:
                    # Test model
                    pred = tf.nn.sigmoid(logits)  # Apply sigmoid to logits
                    correct_prediction = tf.equal(tf.cast(tf.greater(pred, 0.5), tf.float32), y)
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    p1 = pred.eval({x1: test_routes})
                    p2 = tf.cast(tf.greater(p1, 0.5), tf.float32).eval()
                    print("Test Accuracy:", accuracy.eval({x1: test_routes, y: test_successes}))
                    print("Training Accuracy:", accuracy.eval({x1: train_routes, y: train_successes}))
                    print("Time:", datetime.now())



if __name__ == '__main__':
    data = Data()
    simulate = False

    if simulate:
        n_route = 11
        n_position = 5
        n_active = 15
        n_sample = 5200
        n_batch = n_sample/5.
        p_vector= np.random.uniform(low=0.4, high=0.6, size=n_route)
        p_vector = [p_vector[i]/np.sum(p_vector) for i in range(len(p_vector))]

        train_data = data.simulate(n_position=n_position, n_active = n_active, n_sample=n_sample, p_vector=p_vector)

    else:
        train_data = data.real()
        n_route = 11
        n_position = 5
        assert n_route*n_position == train_data['routes'].shape[1], 'number of routes and positions is incorrect, product should be: {}'.format(train_data['routes'].shape[1])

        n_active = train_data['actives'].shape[1]
        n_batch = train_data['actives'].shape[0]/5.

    train(train_data, n_route, n_position, n_active, n_batch, epochs=100000)

    # n_batch = int(np.ceil(n_batch))
    #
    # gibbs_rbm = Gibbs_RBM(n_route=n_route, n_position=n_position, n_active=n_active, n_sample=n_batch)
    #
    # actives = train_data['actives']
    #
    # routes = train_data['routes']
    #
    # for _ in range(10):
    #
    #     active_sample = actives[np.random.choice(actives.shape[0], size=n_batch)].astype(np.float32)
    #
    #     route_sample = routes[np.random.choice(routes.shape[0], size=n_batch)].astype(np.float32)
    #
    #     for _ in range(1):
    #         beta_routes = gibbs_rbm.gibbs_beta_routes_mean(route_sample)
    #         sigma_routes = gibbs_rbm.gibbs_beta_routes_variance(route_sample)
    #
    #         beta_I = gibbs_rbm.gibbs_beta_mean()
    #         sigma_I = gibbs_rbm.gibbs_beta_variance()
    #
    #     out1 = tf.reduce_mean(tf.square(tf.subtract(tf.matmul(gibbs_rbm.w1, tf.transpose(route_sample)), tf.transpose(active_sample))))
    #
    #     _ = gibbs_rbm.learn(0.3, 0.6, 2.1, route_sample, active_sample)
    #
    #     out2 = tf.reduce_mean(tf.square(tf.subtract(tf.matmul(gibbs_rbm.w1, tf.transpose(route_sample)), tf.transpose(active_sample))))
    #
    #     beta_routes = gibbs_rbm.beta_routes
    #     beta_I = gibbs_rbm.beta_I
    #     sigma_routes = gibbs_rbm.sigma_routes
    #     sigma_I = gibbs_rbm.sigma_I
    #
    #     print('hi')









