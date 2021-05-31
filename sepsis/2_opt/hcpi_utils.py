from tqdm import tqdm
from sklearn import utils
import numpy as np

from scipy import stats
from math import log,sqrt


def normalize_return(list_estimate, x_max, x_min=0):
    """
    normalizes the return in [0,1]
    :param list_estimate:
    :return:
    """
    return [(x - x_min) / (x_max - x_min) for x in list_estimate]


def confidence_interval_based(ope_estimator, confidence, pi_t, 
                              R_min=-100, R_max=100,
                              C_min=0, C_max=200,
                              ope_method='DR'):
    """
    """
    # Sec 2.3 of HCPI
    c = 0.5

    # get the estimates, normalize and clip them
    R_list_estimates, C_list_estimates = ope_estimator(pi_t, method=ope_method)
    
    # w.r.t. R
    R_list_estimates = normalize_return(R_list_estimates, x_min=R_min, x_max=R_max, )
    R_list_estimates_cut = [min(x, c) for x in R_list_estimates]
    # subtract.outer => Apply the subtract op to all pairs (a, b) with a in A and b in B.
    R_cross_substract = np.subtract.outer(np.square(R_list_estimates_cut), np.square(R_list_estimates_cut))
    R_cross_substract_squared = np.square(R_cross_substract)

    n = len(R_list_estimates)
    R_lower_bound = (1./n) * np.sum(R_list_estimates_cut) - n/c*sqrt((log(2/confidence))*np.sum(np.sum(R_cross_substract_squared))) - \
                        (7*c*log(2./confidence,2))/(3.*(n-1))

    R_lower_bound = R_lower_bound * (R_max-R_min) + R_min

    # w.r.t. C
    C_list_estimates = normalize_return(C_list_estimates, x_min=C_min, x_max=C_max,)
    C_list_estimates_cut = [min(x, c) for x in C_list_estimates]
    
    C_cross_substract = np.subtract.outer(np.square(C_list_estimates_cut), np.square(C_list_estimates_cut))
    C_cross_substract_squared = np.square(C_cross_substract)
    
    n = len(C_list_estimates)
    C_upper_bound = (1./n) * np.sum(C_list_estimates_cut) + n/c*sqrt((log(2/confidence))*np.sum(np.sum(C_cross_substract_squared))) - \
                        (7*c*log(2./confidence,2))/(3.*(n-1))

    C_upper_bound = C_upper_bound * (C_max-C_min) + C_min
    
    return R_lower_bound, C_upper_bound



def student_t_test(ope_estimator, confidence, pi_t, 
                   R_min=-100, R_max=100,
                   C_min=0, C_max=200,
                   ope_method='DR'):
                        
    # estimate the returns
    R_list_estimates, C_list_estimates = ope_estimator(pi_t, method=ope_method)
    
    # normalize
    R_list_estimates = normalize_return(R_list_estimates, x_min=R_min, x_max=R_max,)
    C_list_estimates = normalize_return(C_list_estimates, x_min=C_min, x_max=C_max,)


    # t-test
    # Sec 2.4 HCPI (Thomas 2015)
    n = len(R_list_estimates)
    assert n == len(C_list_estimates)

    # w.r.t. R
    R_estimated_return = np.mean(R_list_estimates)
    R_sigma = np.sqrt(1./(n-1) * np.sum(np.square(np.array(R_list_estimates) - R_estimated_return)))
    R_lower_bound = R_estimated_return - R_sigma/sqrt((n-1)) * stats.t.ppf(1 - confidence, n-1)
    # undo the normalization here
    R_lower_bound = R_lower_bound * (R_max-R_min) + R_min

    # w.r.t. C    
    C_estimated_return = np.mean(C_list_estimates)
    C_sigma = np.sqrt(1. / (n - 1) * np.sum(np.square(np.array(C_list_estimates) - C_estimated_return)))
    C_upper_bound = C_estimated_return + C_sigma / sqrt((n - 1)) * stats.t.ppf(1 - confidence, n - 1)
    # undo the normalization here
    C_upper_bound = C_upper_bound * (C_max-C_min) + C_min

    return R_lower_bound, C_upper_bound




STRATEGY_DICT = {
    'confidence_interval': confidence_interval_based,
    't_test': student_t_test,
}


def compute_approx_hopt(pi_b, 
                        pi_hat,
                        ope_estimation_on_validation, 
                        confidence, 
                        train_mean_stats, 
                        ope_method = 'DR', 
                        lower_bound_strategy='t_test',
                        C_min=0,
                        C_max=200,
                       ):
    """
    do the HCPI here based on the surrogate pi_hat
    :return:
    """
    
    # pi_hat is the policy that we are going to regularize with baseline

    # ------ Regularize and find the best lower and upper bounds
    
    # Regularization on the optimal policy from previous part
    regularization_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Evaluate all the policy and return the one with the higher lower bound
    # w.r.t R the highest
    best_R_lower_bound = - np.inf
    best_C_upper_bound = + np.inf

    current_best_policy = pi_hat
    current_regularization = 0.0

    # try different linear combinations
    for regularization_parameter in regularization_list:
        # new candidate policy
        current_pi = (1 - regularization_parameter) * pi_hat  + regularization_parameter * pi_b

        # compute the bounds based on the validation
        R_lb, C_ub = STRATEGY_DICT[lower_bound_strategy](ope_estimation_on_validation, confidence, current_pi, 
                                                         ope_method=ope_method, C_max=C_max, C_min=C_min)

    
        # if the bounds are finite, only then use them, else skip
        if (not np.isfinite(R_lb)) or (not np.isfinite(C_ub)):
            continue

        # accept the candidate if it has better performance w.r.t R
        # strictly high R, and lower C
        if R_lb >= best_R_lower_bound and C_ub <= best_C_upper_bound:
            best_policy = current_pi
            best_R_lower_bound = R_lb
            best_C_upper_bound = C_ub
            best_regularization = regularization_parameter

    # -------- Safety Test --------------------
    # check if the best candidate passes the safety test

    pi_hcpi = None
    reg_hcpi = 0.0

    R_perf_baseline = train_mean_stats[0]
    C_perf_baseline = train_mean_stats[1]

    # if passes the safety test 
    if best_R_lower_bound >= R_perf_baseline and best_C_upper_bound <= C_perf_baseline:
        # return the best policy
        pi_hcpi = best_policy
        reg_hcpi = best_regularization
    else:
        # (NSF solution) return the original policy
        pi_hcpi = np.copy(pi_b)
        reg_hcpi = 1.0
    
    assert pi_hcpi is not None
    
    return pi_hcpi, reg_hcpi