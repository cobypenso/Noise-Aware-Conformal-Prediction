import os
import json
import torch
import numpy as np

#######################################################################
################ OURS - NACP ##########################################
#######################################################################
#### raps_inference_only function for inference (meaning given q)
#### NACP_raps function for running NACP

def raps_inference_only(test_probs, test_labels, qhat, lam_reg=0.01, k_reg=5, rand=False, disallow_zero_sets=False):
    
    n = len(test_labels)

    # Deploy
    reg_vec = np.array(k_reg*[0,] + (test_probs.shape[1]-k_reg)*[lam_reg,])[None,:]

    n_test = test_probs.shape[0]
    test_pi = test_probs.argsort(1)[:,::-1]
    test_srt = np.take_along_axis(test_probs,test_pi,axis=1)
    test_srt_reg = test_srt + reg_vec
    test_srt_reg_cumsum = test_srt_reg.cumsum(axis=1)
    indicators = (test_srt_reg.cumsum(axis=1) - np.random.rand(n_test,1)*test_srt_reg) <= qhat if rand else test_srt_reg.cumsum(axis=1) - test_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,test_pi.argsort(axis=1),axis=1)
    
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))

    # Calculate empirical coverage
    empirical_coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]), test_labels
    ].mean()
    print(f"The empirical coverage is: {empirical_coverage}")

    return empirical_coverage, sets


def NACP_raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, noise_rate=0.1, rand=False):
    ### BINARY SEARCH
    n_classes = val_probs.shape[-1]
    n_examples = val_probs.shape[0]

    q_start = 1.0
    q_end = 0.5
    tolerance = 0.00005  # Tolerance for binary search termination
    
    best_q = q_start
    min_diff = float('inf')
    
    while q_start - q_end  > tolerance:  # Continue until the search range is smaller than the tolerance
        qhat = (q_start + q_end) / 2  # Calculate midpoint
        
        A, computed_sets = raps_inference_only(val_probs, val_labels, qhat, rand=rand)

        D = 0
        for idx in range(len(computed_sets)):
            D += (len(computed_sets[idx]) / (n_classes * n_examples))
        
        B = (A - noise_rate * D) / (1 - noise_rate)


        diff = B - (1 - alpha)
        
        if diff > 0:
            # we can further decrease qhat. decrease q_start to be qhat
            q_start = qhat  # Decrease qhat
        else:
            # we can further increase qhat. increase q_end to be qhat
            q_end = qhat  # Increase qhat
        
        # only if diff is positive - i.e coverage holds - check if optimal.
        if (diff < min_diff) and (diff > 0):  # Update best_q if a smaller absolute diff is found
            min_diff = diff
            best_q = qhat
    
    print(f"BestQ: {best_q}, Diff: {min_diff}")
    empirical_coverage, sets = raps_inference_only(test_probs, test_labels, best_q, rand=rand)

    return {
        "qhat": best_q,
        "min_diff": min_diff,
        "empirical_coverage": empirical_coverage,
        "sets": sets
    }


### - baseline - ###################
### based on NRCP git repository ###

def NRCP_raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, lam_reg=0.01, k_reg=5, disallow_zero_sets=False, rand=False, noisy_labels=False, noise_rate=0.2):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels
    n_classes = cal_probs.shape[-1]

    cal_probs_resampled = np.zeros((cal_probs.shape[0] * cal_probs.shape[1], cal_probs.shape[1]))
    cal_labels_resampled = np.zeros((cal_probs.shape[0] * cal_probs.shape[1],), dtype=int)
    weights = np.zeros((cal_probs.shape[0] * cal_probs.shape[1],))
    for idx, (prob, label) in enumerate(zip(cal_probs, cal_labels)):
        for j in range(cal_probs.shape[1]):
            cal_probs_resampled[(idx * n_classes) + j, :] = prob
            cal_labels_resampled[(idx * n_classes) + j] = j
            if j == label:
                weights[(idx * n_classes) + j] = (1 - noise_rate)
            else:
                weights[(idx * n_classes) + j] = noise_rate / (n_classes - 1)

    n = len(cal_labels_resampled)

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    reg_vec = np.array(k_reg*[0,] + (cal_probs_resampled.shape[1]-k_reg)*[lam_reg,])[None,:]
    cal_pi = cal_probs_resampled.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_probs_resampled,cal_pi,axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels_resampled[:,None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n),cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n),cal_L]

    cal_scores_weighted =  cal_scores * weights
    res = cal_scores_weighted.reshape(-1,n_classes).sum(axis = 1)
    # Get the score quantile
    qhat = np.quantile(res, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')

    # Deploy
    n_val = val_probs.shape[0]
    val_pi = val_probs.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_probs,val_pi,axis=1)
    val_srt_reg = val_srt + reg_vec
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
        
    fixed_qhat = (qhat - noise_rate * val_srt.mean(axis=-1)) / (1 - noise_rate)
    indicators_with_fixed_qhat = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= fixed_qhat[:, np.newaxis] if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= fixed_qhat[:, np.newaxis]
    if disallow_zero_sets: indicators_with_fixed_qhat[:,0] = True
    prediction_sets_with_fixed_qhat = np.take_along_axis(indicators_with_fixed_qhat,val_pi.argsort(axis=1),axis=1)

    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))

    sets_with_fixed_qhat = []
    for i in range(len(prediction_sets)):
        sets_with_fixed_qhat.append(tuple(np.where(prediction_sets_with_fixed_qhat[i, :] != 0)[0]))

    return (sets, sets_with_fixed_qhat, val_labels), qhat, fixed_qhat