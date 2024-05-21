import os
import json
import torch
import numpy as np

# ---- imports ---- #
from HPS import hps
from APS import aps, aps_randomized
from RAPS import raps
from nacp_APS import NACP_aps, NACP_aps_randomized
from nacp_RAPS import NACP_raps
from nacp_HPS import NACP_hps



def calc_baseline_mets(val_probs, val_labels, test_probs, test_labels, n_calib=0, alpha=0.1,
                       model_names=['aps', 'NACP_aps'],
                       k_raps=5, noise_rate = 0.2):
    mets = {}

    ##########################################################################
    ################ Original Conformal prediction ###########################
    ##########################################################################


    if 'hps' in model_names:
        (sets, labels), qhat = hps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha)
        mets['hps'] = calc_conformal_mets(sets, labels)
        mets['hps_qhat'] = qhat
        print ('hps: ----> qhat:', qhat)
        print ('hps: ----> mean_set_size:', mets['hps']['size_mean'])

    if 'aps' in model_names:
        (sets, labels), qhat = aps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha)
        mets['aps'] = calc_conformal_mets(sets, labels)
        mets['aps_qhat'] = qhat
        print ('aps: ----> qhat:', qhat)
        print ('aps: ----> mean_set_size:', mets['aps']['size_mean'])

    if 'aps_randomized' in model_names:
        (sets, labels), qhat = aps_randomized(val_probs, val_labels, test_probs, test_labels, n_calib, alpha)
        mets['aps_randomized'] = calc_conformal_mets(sets, labels)
        mets['aps_randomized_qhat'] = qhat
        print ('aps_randomized: ----> qhat:', qhat)
        print ('aps_randomized: ----> mean_set_size:', mets['aps_randomized']['size_mean'])

    if 'raps' in model_names:
        (sets, labels), qhat = raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, rand=False, k_reg=k_raps)
        mets['raps'] = calc_conformal_mets(sets, labels)
        mets['raps_qhat'] = qhat
        print ('raps: ----> qhat:', qhat)
        print ('raps: ----> mean_set_size:', mets['raps']['size_mean'])
    
    if 'raps_randomized' in model_names:
        (sets, labels), qhat = raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, rand=True, k_reg=k_raps)
        mets['raps_randomized'] = calc_conformal_mets(sets, labels)
        mets['raps_randomized_qhat'] = qhat

    ##########################################################################
    ############### Noise Aware Conformal prediction #########################
    ##########################################################################

    if 'NACP_hps' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = NACP_hps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, noise_rate = noise_rate)
        mets['NACP_hps'] = calc_conformal_mets(sets, labels)
        mets['NACP_hps_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['NACP_hps_qhat'] = qhat
        mets['NACP_hps_with_fixed_qhat_qhat'] = fixed_qhat
        print ('NACP_hps: ----> qhat:', qhat)
        print ('NACP_hps: ----> mean_set_size:', mets['NACP_hps']['size_mean'])

    if 'NACP_aps' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = NACP_aps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, noise_rate = noise_rate)
        mets['NACP_aps'] = calc_conformal_mets(sets, labels)
        mets['NACP_aps_qhat'] = qhat
        mets['NACP_aps_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['NACP_aps_with_fixed_qhat_qhat'] = fixed_qhat
        print ('NACP_aps: ----> qhat:', qhat)
        print ('NACP_aps: ----> mean_set_size:', mets['NACP_aps']['size_mean'])
    

    if 'NACP_aps_randomized' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = NACP_aps_randomized(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, noise_rate = noise_rate)
        mets['NACP_aps_randomized'] = calc_conformal_mets(sets, labels)
        mets['NACP_aps_randomized_qhat'] = qhat
        mets['NACP_aps_randomized_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['NACP_aps_randomized_with_fixed_qhat_qhat'] = fixed_qhat
        print ('NACP_aps_randomized: ----> qhat:', qhat)
        print ('NACP_aps_randomized: ----> mean_set_size:', mets['NACP_aps_randomized']['size_mean'])
    
    if 'NACP_raps' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = NACP_raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, rand=False, k_reg=k_raps, noise_rate = noise_rate)
        mets['NACP_raps'] = calc_conformal_mets(sets, labels)
        mets['NACP_raps_qhat'] = qhat
        mets['NACP_raps_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['NACP_raps_with_fixed_qhat_qhat'] = fixed_qhat
        print ('NACP_raps: ----> qhat:', qhat)
        print ('NACP_raps: ----> mean_set_size:', mets['NACP_raps']['size_mean'])

    if 'NACP_raps_randomized' in model_names:
        (sets, sets_with_fixed_qhat, labels), qhat, fixed_qhat = NACP_raps(val_probs, val_labels, test_probs, test_labels, n_calib, alpha, rand=True, k_reg=k_raps, noise_rate = noise_rate)
        mets['NACP_raps_randomized'] = calc_conformal_mets(sets, labels)
        mets['NACP_raps_randomized_qhat'] = qhat
        mets['NACP_raps_randomized_with_fixed_qhat'] = calc_conformal_mets(sets_with_fixed_qhat, labels)
        mets['NACP_raps_randomized_with_fixed_qhat_qhat'] = fixed_qhat
        print ('NACP_raps_randomized: ----> qhat:', qhat)
        print ('NACP_raps_randomized: ----> mean_set_size:', mets['NACP_raps_randomized']['size_mean'])

    return mets


def calc_conformal_mets(sets, labels):
    set_lens = []
    hits = []
    hits_per_label = {}
    for s, l in zip(sets, labels):
        set_lens.append(len(s))
        if l not in hits_per_label:
            hits_per_label[l] = []
        if l in s:
            hits.append(1)
            hits_per_label[l].append(1)
        else:
            hits.append(0)
            hits_per_label[l].append(0)

    acc = np.asarray(hits).mean()
    set_lens = np.asarray(set_lens)
    acc_per_label = {k: np.mean(v) for k, v in hits_per_label.items()}
    acc_per_label = np.asarray(list(acc_per_label.values()))
    return {'size_mean': set_lens.mean(),
            'size_std': set_lens.std(),
            'acc': acc}