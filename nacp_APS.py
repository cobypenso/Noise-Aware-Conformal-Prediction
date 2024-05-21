import os
import json
import torch
import numpy as np

#######################################################################
################ OURS - NACP ##########################################
#######################################################################
#### hps_inference_only function for inference (meaning given q)
#### NACP_hps function for running NACP


def aps_inference_only(test_probs, test_labels, qhat):
    
    n = len(test_labels)

    # Deploy (output=list of length n, each element is tensor of classes)
    test_pi = test_probs.argsort(1)[:, ::-1]
    test_srt = np.take_along_axis(test_probs, test_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(test_srt <= qhat, test_pi.argsort(axis=1), axis=1)

    sets = [] 
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))

    # Calculate empirical coverage
    empirical_coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]), test_labels
    ].mean()
    print(f"The empirical coverage is: {empirical_coverage}")

    return empirical_coverage, sets

def NACP_aps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, noise_rate=0.1):
    """
        noise_format - "all" or "rest".
        "all" == (1-eps){i=j} + eps / k
        "rest" == (1-eps){i=j} + (eps / (k -1)){i!=j}
    """
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
        
        A, computed_sets = aps_inference_only(val_probs, val_labels, qhat)

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
    empirical_coverage, sets = aps_inference_only(test_probs, test_labels, best_q)

    return {
        "qhat": best_q,
        "min_diff": min_diff,
        "empirical_coverage": empirical_coverage,
        "sets": sets
    }

def aps_randomized_inference_only(test_probs, test_labels, qhat, no_zero_size_sets=False):

    # Deploy (output=list of length n, each element is tensor of classes)
    test_pi = test_probs.argsort(1)[:, ::-1]
    test_srt = np.take_along_axis(test_probs, test_pi, axis=1).cumsum(axis=1)

    n_test = test_srt.shape[0]
    cumsum_index = np.sum(test_srt <= qhat, axis=1)
    high = test_srt[np.arange(n_test), cumsum_index]
    low = np.zeros_like(high)
    low[cumsum_index > 0] = test_srt[np.arange(n_test), cumsum_index-1][cumsum_index > 0]
    prob = (qhat - low)/(high - low)
    rv = np.random.binomial(1,prob,size=(n_test))
    randomized_threshold = low 
    randomized_threshold[rv == 1] = high[rv == 1]
    if no_zero_size_sets:
        randomized_threshold = np.maximum(randomized_threshold, test_srt[:,0])
    prediction_sets = np.take_along_axis(test_srt <= randomized_threshold[:,None], test_pi.argsort(axis=1), axis=1)
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    
    # Calculate empirical coverage
    empirical_coverage = prediction_sets[
        np.arange(prediction_sets.shape[0]), test_labels
    ].mean()

    return empirical_coverage, sets



def NACP_aps_randomized(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, noise_rate=0.1):
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
        
        A, computed_sets = aps_randomized_inference_only(val_probs, val_labels, qhat)

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
    empirical_coverage, sets = aps_randomized_inference_only(test_probs, test_labels, best_q)

    return {
        "qhat": best_q,
        "min_diff": min_diff,
        "empirical_coverage": empirical_coverage,
        "sets": sets
    }

### - baseline - ###################
### based on NRCP git repository ###

def NRCP_aps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, noise_rate = 0.1):
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

    # sort cal_probs based on axis = 1, and then reverse the order on axis = 1
    cal_pi = cal_probs_resampled.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_probs_resampled, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels_resampled
    ]
    cal_scores_weighted =  cal_scores * weights
    res = cal_scores_weighted.reshape(-1,n_classes).sum(axis = 1)
    qhat = np.quantile(
        res, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )

    # Deploy (output=list of length n, each element is tensor of classes)
    val_pi = val_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)

    fixed_qhat = (qhat - noise_rate * val_srt.mean(axis=-1)) / (1 - noise_rate)
    prediction_sets_with_fixed_qhat = np.take_along_axis(val_srt <= fixed_qhat[:, np.newaxis], val_pi.argsort(axis=1), axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    

    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))

    sets_with_fixed_qhat = []
    for i in range(len(prediction_sets_with_fixed_qhat)):
        sets_with_fixed_qhat.append(tuple(np.where(prediction_sets_with_fixed_qhat[i, :] != 0)[0]))

    return (sets, sets_with_fixed_qhat, val_labels), qhat, fixed_qhat




def NRCP_aps_randomized(val_probs, val_labels, test_probs, test_labels, n_calib, alpha=0.1, randomized=True, no_zero_size_sets=False, noise_rate = 0.1):
    cal_probs, cal_labels, val_probs, val_labels = val_probs, val_labels, test_probs, test_labels

    n_classes = cal_probs.shape[-1]
    n_examples = cal_probs.shape[0] 

    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    cal_probs = cal_probs.astype(np.float64)

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

    cal_pi = cal_probs_resampled.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_probs_resampled, cal_pi, axis=1).cumsum(axis=1)
    cal_softmax_correct_class = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels_resampled
    ]
    if not randomized:
        cal_scores = cal_softmax_correct_class
    else:
        cumsum_index = np.where(cal_srt == cal_softmax_correct_class[:,None])[1]
        if cumsum_index.shape[0] != cal_srt.shape[0]:
            _, unique_indices = np.unique(np.where(
                cal_srt == cal_softmax_correct_class[:,None])[0], return_index=True)
            cumsum_index = cumsum_index[unique_indices]

        high = cal_softmax_correct_class
        low = np.zeros_like(high)
        low[cumsum_index != 0] = cal_srt[np.where(cumsum_index != 0)[0], cumsum_index[cumsum_index != 0]-1]
        cal_scores = np.random.uniform(low=low, high=high)

    cal_scores_weighted =  cal_scores * weights
    res = cal_scores_weighted.reshape(-1,n_classes).sum(axis = 1)

    # Get the score quantile
    qhat = np.quantile(
        res, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
    )

    # Deploy (output=list of length n, each element is tensor of classes)
    val_pi = val_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_probs, val_pi, axis=1).cumsum(axis=1)
    if not randomized:
        fixed_qhat = (qhat - noise_rate * val_srt.mean(axis=-1)) / (1 - noise_rate)
        prediction_sets_with_fixed_qhat = np.take_along_axis(val_srt <= fixed_qhat[:, np.newaxis], val_pi.argsort(axis=1), axis=1)
        prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    else:
        n_val = val_srt.shape[0]
        cumsum_index = np.sum(val_srt <= qhat, axis=1)
        high = val_srt[np.arange(n_val), cumsum_index]
        low = np.zeros_like(high)
        low[cumsum_index > 0] = val_srt[np.arange(n_val), cumsum_index-1][cumsum_index > 0]
        prob = (qhat - low)/(high - low)
        rv = np.random.binomial(1,prob,size=(n_val))
        randomized_threshold = low
        randomized_threshold[rv == 1] = high[rv == 1]
        if no_zero_size_sets:
            randomized_threshold = np.maximum(randomized_threshold, val_srt[:,0])
        
        fixed_qhat = (randomized_threshold - noise_rate * val_srt.mean(axis=-1)) / (1 - noise_rate)
        prediction_sets_with_fixed_qhat = np.take_along_axis(val_srt <= fixed_qhat[:, np.newaxis], val_pi.argsort(axis=1), axis=1)
        prediction_sets = np.take_along_axis(val_srt <= randomized_threshold[:,None], val_pi.argsort(axis=1), axis=1)
    
    sets = []
    for i in range(len(prediction_sets)):
        sets.append(tuple(np.where(prediction_sets[i, :] != 0)[0]))
    
    sets_with_fixed_qhat = []
    for i in range(len(prediction_sets_with_fixed_qhat)):
        sets_with_fixed_qhat.append(tuple(np.where(prediction_sets_with_fixed_qhat[i, :] != 0)[0]))

    return (sets, sets_with_fixed_qhat, val_labels), qhat, fixed_qhat