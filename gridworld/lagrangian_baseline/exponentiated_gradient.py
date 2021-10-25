import numpy as np

class ExponentiatedGradient(object):
    """
    Extended from: https://github.com/clvoloshin/constrained_batch_policy_learning/blob/master/exponentiated_gradient.py

    """
    def __init__(self, lambda_bound, number_of_constraints, eta=1., starting_lambda='uniform'):
        '''
        lambda_bound: hyper-param B in Alg 2
        eta: hyper-param \eta in Alg 2
        numb_constraints = (# of objectives) + 1
            additional 1 added for the phantom constraint for bounded norm (Sec 3, Le et al)
        '''
        self.eta = eta
        self.lambda_bound = lambda_bound
        self.number_of_constraints = number_of_constraints
        if starting_lambda == 'uniform':
            self.w_t = self.lambda_bound * np.ones(self.number_of_constraints) / self.number_of_constraints
        else:
            self.w_t = starting_lambda
            self.lambda_bound = np.sum(starting_lambda)

    def run(self, gradient):
        """
        This function implements Line 16 of Alg 2

        :param gradient: vector z_t
        Z_t[i]: of the form (J(i) <= tau)

        for maximizatio
        :return:
        """
        self.w_t = self.w_t/self.lambda_bound

        unnormalized_wt = self.w_t*np.exp(self.eta*gradient) # positive since working  w/ costs.
        self.w_t = self.lambda_bound*unnormalized_wt/sum(unnormalized_wt)
        return self.w_t


    def get(self):
        return self.w_t