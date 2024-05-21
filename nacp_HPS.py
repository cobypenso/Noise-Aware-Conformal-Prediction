import os
import json
import torch
import numpy as np

#######################################################################
################ OURS - NACP ##########################################
#######################################################################
#### hps_inference_only function for inference (meaning given q)
#### NACP_hps function for running NACP

def hps_inference_only(test_probs, test_labels, qhat):
    prediction_sets = test_probs >= (1 - qhat)
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    
    # Calculate empirical coverage
    empirical_coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]), test_labels
    ].mean()
    print(f"The empirical coverage is: {empirical_coverage}")

    return empirical_coverage, sets

def NACP_hps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, noise_rate=0.2):
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
        
        A, computed_sets = hps_inference_only(val_probs, val_labels, qhat)

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
    empirical_coverage, sets = hps_inference_only(test_probs, test_labels, best_q)

    return {
        "qhat": best_q,
        "min_diff": min_diff,
        "empirical_coverage": empirical_coverage,
        "sets": sets
    }


### - baseline - ###################
### based on NRCP git repository ###

def NRCP_hps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, noise_rate = 0.1):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels
    n_classes = cal_probs.shape[-1]
    n_examples = cal_probs.shape[0]
    
    # 0: compute noise-robust scores - weighted average based on noise rate
    cal_probs_resampled = np.zeros((n_examples * n_classes, n_classes))
    cal_labels_resampled = np.zeros((n_examples * n_classes,), dtype=int)
    weights = np.zeros((n_examples * n_classes,))
    for idx, (prob, label) in enumerate(zip(cal_probs, cal_labels)):
        for j in range(n_classes):
            cal_probs_resampled[(idx * n_classes) + j, :] = prob
            cal_labels_resampled[(idx * n_classes) + j] = j
            if j == label:
                weights[(idx * n_classes) + j] = (1 - noise_rate)
            else:
                weights[(idx * n_classes) + j] = (noise_rate / (n_classes - 1))

    n = len(cal_labels_resampled)
    
    # 1: get conformal scores. n = calib_Y.shape[0]
    cal_scores = 1 - cal_probs_resampled[np.arange(n), cal_labels_resampled]
    # 2: get adjusted quantile - based on original n
    q_level = np.ceil((n + 1) * (1 - alpha)) / n

    # weight and average based on noise level
    cal_scores_weighted =  cal_scores * weights
    res = cal_scores_weighted.reshape(-1,n_classes).sum(axis = 1)

    # compute qhat
    qhat = np.quantile(res, q_level, interpolation="higher")

    prediction_sets = val_probs >= (1 - qhat)
    
    fixed_qhat = (qhat - noise_rate * (1-val_probs).mean(axis=-1)) / (1 - noise_rate)
    prediction_sets_with_fixed_qhat = val_probs >= (1 - fixed_qhat[:, np.newaxis])

    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    
    sets_with_fixed_qhat = []
    for i in range(len(prediction_sets)):
        sets_with_fixed_qhat.append(tuple(np.where(prediction_sets_with_fixed_qhat[i, :] != 0)[0]))

    return (sets, sets_with_fixed_qhat, val_labels), qhat, fixed_qhat