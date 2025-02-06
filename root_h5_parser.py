import uproot
import numpy as np
import awkward as ak
import vector
import h5py 
import sys
from numpy.random import seed
seed_value=420
seed(seed_value)
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import argparse


def duplicate_cleaner(matched_pairs, truth=None):
    truth_ids = matched_pairs['truth']['id']
    deduped_id_pairs = ak.Array([list(dict.fromkeys(row)) for row in ak.to_list(truth_ids)])
    no_repeats = ak.num(deduped_id_pairs) == ak.num(truth_ids)
    one_repeat = ak.num(deduped_id_pairs) == ak.num(truth_ids) - 1

    duplicate_pairs = matched_pairs[one_repeat]
    clean_pairs = matched_pairs[no_repeats]
    if truth is not None:
        duplicate_truth = truth[one_repeat]
        return clean_pairs, duplicate_pairs, duplicate_truth
    
    return clean_pairs, duplicate_pairs, no_repeats

def top_identifier(matched_data, process, lepton_channel, lepton_indices=None):
    """
    Takes the matched, duplicate free data and trys to reconstruct any tops it can
    """

    if process=="4t":
        if lepton_channel == "0L":
            truth_tops_1 = matched_data['truth']['truth'][(matched_data['truth']['id']==1)|(matched_data['truth']['id']==5)|(matched_data['truth']['id']==6)]
            truth_tops_2 = matched_data['truth']['truth'][(matched_data['truth']['id']==2)|(matched_data['truth']['id']==7)|(matched_data['truth']['id']==8)]
            truth_tops_3 = matched_data['truth']['truth'][(matched_data['truth']['id']==3)|(matched_data['truth']['id']==9)|(matched_data['truth']['id']==10)]
            truth_tops_4 = matched_data['truth']['truth'][(matched_data['truth']['id']==4)|(matched_data['truth']['id']==11)|(matched_data['truth']['id']==12)]

            reco_tops_1 = matched_data['reco'][(matched_data['truth']['id']==1)|(matched_data['truth']['id']==5)|(matched_data['truth']['id']==6)]
            reco_tops_2 = matched_data['reco'][(matched_data['truth']['id']==2)|(matched_data['truth']['id']==7)|(matched_data['truth']['id']==8)]
            reco_tops_3 = matched_data['reco'][(matched_data['truth']['id']==3)|(matched_data['truth']['id']==9)|(matched_data['truth']['id']==10)]
            reco_tops_4 = matched_data['reco'][(matched_data['truth']['id']==4)|(matched_data['truth']['id']==11)|(matched_data['truth']['id']==12)]

            truth_tops_1 = ak.mask(truth_tops_1, ak.num(truth_tops_1)==3)
            truth_tops_2 = ak.mask(truth_tops_2, ak.num(truth_tops_2)==3)
            truth_tops_3 = ak.mask(truth_tops_3, ak.num(truth_tops_3)==3)
            truth_tops_4 = ak.mask(truth_tops_4, ak.num(truth_tops_4)==3)

            reco_tops_1 = ak.mask(reco_tops_1, ak.num(reco_tops_1)==3)
            reco_tops_2 = ak.mask(reco_tops_2, ak.num(reco_tops_2)==3)
            reco_tops_3 = ak.mask(reco_tops_3, ak.num(reco_tops_3)==3)
            reco_tops_4 = ak.mask(reco_tops_4, ak.num(reco_tops_4)==3)

        elif lepton_channel == "2L":

            matched_01 = matched_data[(lepton_indices[:,0]==0)&(lepton_indices[:,1]==1)]
            truth_tops_1_01 = matched_01['truth']['truth'][(matched_01['truth']['id']==1)|(matched_01['truth']['id']==5)]
            truth_tops_2_01 = matched_01['truth']['truth'][(matched_01['truth']['id']==2)|(matched_01['truth']['id']==6)]
            truth_tops_3_01 = matched_01['truth']['truth'][(matched_01['truth']['id']==3)|(matched_01['truth']['id']==7)|(matched_01['truth']['id']==8)]
            truth_tops_4_01 = matched_01['truth']['truth'][(matched_01['truth']['id']==4)|(matched_01['truth']['id']==9)|(matched_01['truth']['id']==10)]

            reco_tops_1_01 = matched_01['reco'][(matched_01['truth']['id']==1)|(matched_01['truth']['id']==5)]
            reco_tops_2_01 = matched_01['reco'][(matched_01['truth']['id']==2)|(matched_01['truth']['id']==6)]
            reco_tops_3_01 = matched_01['reco'][(matched_01['truth']['id']==3)|(matched_01['truth']['id']==7)|(matched_01['truth']['id']==8)]
            reco_tops_4_01 = matched_01['reco'][(matched_01['truth']['id']==4)|(matched_01['truth']['id']==9)|(matched_01['truth']['id']==10)]

            matched_03 = matched_data[(lepton_indices[:,0]==0)&(lepton_indices[:,1]==3)]
            truth_tops_1_03 = matched_03['truth']['truth'][(matched_03['truth']['id']==1)|(matched_03['truth']['id']==5)]
            truth_tops_3_03 = matched_03['truth']['truth'][(matched_03['truth']['id']==2)|(matched_03['truth']['id']==6)|(matched_03['truth']['id']==7)]
            truth_tops_2_03 = matched_03['truth']['truth'][(matched_03['truth']['id']==3)|(matched_03['truth']['id']==8)]
            truth_tops_4_03 = matched_03['truth']['truth'][(matched_03['truth']['id']==4)|(matched_03['truth']['id']==9)|(matched_03['truth']['id']==10)]

            reco_tops_1_03 = matched_03['reco'][(matched_03['truth']['id']==1)|(matched_03['truth']['id']==5)]
            reco_tops_3_03 = matched_03['reco'][(matched_03['truth']['id']==2)|(matched_03['truth']['id']==6)|(matched_03['truth']['id']==7)]
            reco_tops_2_03 = matched_03['reco'][(matched_03['truth']['id']==3)|(matched_03['truth']['id']==8)]
            reco_tops_4_03 = matched_03['reco'][(matched_03['truth']['id']==4)|(matched_03['truth']['id']==9)|(matched_03['truth']['id']==10)]

            matched_05 = matched_data[(lepton_indices[:,0]==0)&(lepton_indices[:,1]==5)]
            truth_tops_1_05 = matched_05['truth']['truth'][(matched_05['truth']['id']==1)|(matched_05['truth']['id']==5)]
            truth_tops_3_05 = matched_05['truth']['truth'][(matched_05['truth']['id']==2)|(matched_05['truth']['id']==6)|(matched_05['truth']['id']==7)]
            truth_tops_4_05 = matched_05['truth']['truth'][(matched_05['truth']['id']==3)|(matched_05['truth']['id']==8)|(matched_05['truth']['id']==9)]
            truth_tops_2_05 = matched_05['truth']['truth'][(matched_05['truth']['id']==4)|(matched_05['truth']['id']==10)]

            reco_tops_1_05 = matched_05['reco'][(matched_05['truth']['id']==1)|(matched_05['truth']['id']==5)]
            reco_tops_3_05 = matched_05['reco'][(matched_05['truth']['id']==2)|(matched_05['truth']['id']==6)|(matched_05['truth']['id']==7)]
            reco_tops_4_05 = matched_05['reco'][(matched_05['truth']['id']==3)|(matched_05['truth']['id']==8)|(matched_05['truth']['id']==9)]
            reco_tops_2_05 = matched_05['reco'][(matched_05['truth']['id']==4)|(matched_05['truth']['id']==10)]

            matched_23 = matched_data[(lepton_indices[:,0]==2)&(lepton_indices[:,1]==3)]
            truth_tops_3_23 = matched_23['truth']['truth'][(matched_23['truth']['id']==1)|(matched_23['truth']['id']==5)|(matched_23['truth']['id']==6)]
            truth_tops_1_23 = matched_23['truth']['truth'][(matched_23['truth']['id']==2)|(matched_23['truth']['id']==7)]
            truth_tops_2_23 = matched_23['truth']['truth'][(matched_23['truth']['id']==3)|(matched_23['truth']['id']==8)]
            truth_tops_4_23 = matched_23['truth']['truth'][(matched_23['truth']['id']==4)|(matched_23['truth']['id']==9)|(matched_23['truth']['id']==10)]

            reco_tops_3_23 = matched_23['reco'][(matched_23['truth']['id']==1)|(matched_23['truth']['id']==5)|(matched_23['truth']['id']==6)]
            reco_tops_1_23 = matched_23['reco'][(matched_23['truth']['id']==2)|(matched_23['truth']['id']==7)]
            reco_tops_2_23 = matched_23['reco'][(matched_23['truth']['id']==3)|(matched_23['truth']['id']==8)]
            reco_tops_4_23 = matched_23['reco'][(matched_23['truth']['id']==4)|(matched_23['truth']['id']==9)|(matched_23['truth']['id']==10)]

            matched_25 = matched_data[(lepton_indices[:,0]==2)&(lepton_indices[:,1]==5)]
            truth_tops_3_25 = matched_25['truth']['truth'][(matched_25['truth']['id']==1)|(matched_25['truth']['id']==5)|(matched_25['truth']['id']==6)]
            truth_tops_1_25 = matched_25['truth']['truth'][(matched_25['truth']['id']==2)|(matched_25['truth']['id']==7)]
            truth_tops_4_25 = matched_25['truth']['truth'][(matched_25['truth']['id']==3)|(matched_25['truth']['id']==8)|(matched_25['truth']['id']==9)]
            truth_tops_2_25 = matched_25['truth']['truth'][(matched_25['truth']['id']==4)|(matched_25['truth']['id']==10)]

            reco_tops_3_25 = matched_25['reco'][(matched_25['truth']['id']==1)|(matched_25['truth']['id']==5)|(matched_25['truth']['id']==6)]
            reco_tops_1_25 = matched_25['reco'][(matched_25['truth']['id']==2)|(matched_25['truth']['id']==7)]
            reco_tops_4_25 = matched_25['reco'][(matched_25['truth']['id']==3)|(matched_25['truth']['id']==8)|(matched_25['truth']['id']==9)]
            reco_tops_2_25 = matched_25['reco'][(matched_25['truth']['id']==4)|(matched_25['truth']['id']==10)]

            matched_45 = matched_data[lepton_indices[:,0]==4]
            truth_tops_3_45 = matched_45['truth']['truth'][(matched_45['truth']['id']==1)|(matched_45['truth']['id']==5)|(matched_45['truth']['id']==6)]
            truth_tops_4_45 = matched_45['truth']['truth'][(matched_45['truth']['id']==2)|(matched_45['truth']['id']==7)|(matched_45['truth']['id']==8)]
            truth_tops_1_45 = matched_45['truth']['truth'][(matched_45['truth']['id']==3)|(matched_45['truth']['id']==9)]
            truth_tops_2_45 = matched_45['truth']['truth'][(matched_45['truth']['id']==4)|(matched_45['truth']['id']==10)]

            reco_tops_3_45 = matched_45['reco'][(matched_45['truth']['id']==1)|(matched_45['truth']['id']==5)|(matched_45['truth']['id']==6)]
            reco_tops_4_45 = matched_45['reco'][(matched_45['truth']['id']==2)|(matched_45['truth']['id']==7)|(matched_45['truth']['id']==8)]
            reco_tops_1_45 = matched_45['reco'][(matched_45['truth']['id']==3)|(matched_45['truth']['id']==9)]
            reco_tops_2_45 = matched_45['reco'][(matched_45['truth']['id']==4)|(matched_45['truth']['id']==10)]

            truth_tops_1 = ak.concatenate((truth_tops_1_01, truth_tops_1_03, truth_tops_1_05, truth_tops_1_23, truth_tops_1_25, truth_tops_1_45))
            truth_tops_2 = ak.concatenate((truth_tops_2_01, truth_tops_2_03, truth_tops_2_05, truth_tops_2_23, truth_tops_2_25, truth_tops_2_45))
            truth_tops_3 = ak.concatenate((truth_tops_3_01, truth_tops_3_03, truth_tops_3_05, truth_tops_3_23, truth_tops_3_25, truth_tops_3_45))
            truth_tops_4 = ak.concatenate((truth_tops_4_01, truth_tops_4_03, truth_tops_4_05, truth_tops_4_23, truth_tops_4_25, truth_tops_4_45))

            reco_tops_1 = ak.concatenate((reco_tops_1_01, reco_tops_1_03, reco_tops_1_05, reco_tops_1_23, reco_tops_1_25, reco_tops_1_45))
            reco_tops_2 = ak.concatenate((reco_tops_2_01, reco_tops_2_03, reco_tops_2_05, reco_tops_2_23, reco_tops_2_25, reco_tops_2_45))
            reco_tops_3 = ak.concatenate((reco_tops_3_01, reco_tops_3_03, reco_tops_3_05, reco_tops_3_23, reco_tops_3_25, reco_tops_3_45))
            reco_tops_4 = ak.concatenate((reco_tops_4_01, reco_tops_4_03, reco_tops_4_05, reco_tops_4_23, reco_tops_4_25, reco_tops_4_45))

            truth_tops_1 = ak.mask(truth_tops_1, ak.num(truth_tops_1)==2)
            truth_tops_2 = ak.mask(truth_tops_2, ak.num(truth_tops_2)==2)
            truth_tops_3 = ak.mask(truth_tops_3, ak.num(truth_tops_3)==3)
            truth_tops_4 = ak.mask(truth_tops_4, ak.num(truth_tops_4)==3)

            reco_tops_1 = ak.mask(reco_tops_1, ak.num(reco_tops_1)==2)
            reco_tops_2 = ak.mask(reco_tops_2, ak.num(reco_tops_2)==2)
            reco_tops_3 = ak.mask(reco_tops_3, ak.num(reco_tops_3)==3)
            reco_tops_4 = ak.mask(reco_tops_4, ak.num(reco_tops_4)==3)

        truth_top_1 = ak.unflatten(ak.sum(truth_tops_1, axis=1), 1)
        truth_top_2 = ak.unflatten(ak.sum(truth_tops_2, axis=1), 1)
        truth_top_3 = ak.unflatten(ak.sum(truth_tops_3, axis=1), 1)
        truth_top_4 = ak.unflatten(ak.sum(truth_tops_4, axis=1), 1)

        reco_top_1 = ak.unflatten(ak.sum(reco_tops_1, axis=1), 1)
        reco_top_2 = ak.unflatten(ak.sum(reco_tops_2, axis=1), 1)
        reco_top_3 = ak.unflatten(ak.sum(reco_tops_3, axis=1), 1)
        reco_top_4 = ak.unflatten(ak.sum(reco_tops_4, axis=1), 1)

        all_truth_tops = ak.drop_none(ak.concatenate((truth_top_1,truth_top_2,truth_top_3,truth_top_4), axis=1))
        all_reco_tops = ak.drop_none(ak.concatenate((reco_top_1,reco_top_2,reco_top_3,reco_top_4), axis=1))

        sm_mask, dm_mask, tm_mask, fm_mask = (ak.num(all_truth_tops)==1), (ak.num(all_truth_tops)==2), (ak.num(all_truth_tops)==3), (ak.num(all_truth_tops)==4)

        sm_truth_tops, sm_reco_tops = all_truth_tops[sm_mask], all_reco_tops[sm_mask]
        dm_truth_tops, dm_reco_tops = all_truth_tops[dm_mask], all_reco_tops[dm_mask]
        tm_truth_tops, tm_reco_tops = all_truth_tops[tm_mask], all_reco_tops[tm_mask]
        fm_truth_tops, fm_reco_tops = all_truth_tops[fm_mask], all_reco_tops[fm_mask]

        return [sm_truth_tops, sm_reco_tops], [dm_truth_tops, dm_reco_tops], [tm_truth_tops, tm_reco_tops], [fm_truth_tops, fm_reco_tops]
    elif (process=="3tj")|(process=="3tW"):
        if lepton_channel == "0L":
            truth_tops_1 = matched_data['truth']['truth'][(matched_data['truth']['id']==1)|(matched_data['truth']['id']==4)|(matched_data['truth']['id']==5)]
            truth_tops_2 = matched_data['truth']['truth'][(matched_data['truth']['id']==2)|(matched_data['truth']['id']==6)|(matched_data['truth']['id']==7)]
            truth_tops_3 = matched_data['truth']['truth'][(matched_data['truth']['id']==3)|(matched_data['truth']['id']==8)|(matched_data['truth']['id']==9)]

            reco_tops_1 = matched_data['reco'][(matched_data['truth']['id']==1)|(matched_data['truth']['id']==4)|(matched_data['truth']['id']==5)]
            reco_tops_2 = matched_data['reco'][(matched_data['truth']['id']==2)|(matched_data['truth']['id']==6)|(matched_data['truth']['id']==7)]
            reco_tops_3 = matched_data['reco'][(matched_data['truth']['id']==3)|(matched_data['truth']['id']==8)|(matched_data['truth']['id']==9)]

            truth_tops_1 = ak.mask(truth_tops_1, ak.num(truth_tops_1)==3)
            truth_tops_2 = ak.mask(truth_tops_2, ak.num(truth_tops_2)==3)
            truth_tops_3 = ak.mask(truth_tops_3, ak.num(truth_tops_3)==3)

            reco_tops_1 = ak.mask(reco_tops_1, ak.num(reco_tops_1)==3)
            reco_tops_2 = ak.mask(reco_tops_2, ak.num(reco_tops_2)==3)
            reco_tops_3 = ak.mask(reco_tops_3, ak.num(reco_tops_3)==3)

        elif lepton_channel == "2L":

            matched_01 = matched_data[(lepton_indices[:,0]==0)&(lepton_indices[:,1]==1)]
            truth_tops_1_01 = matched_01['truth']['truth'][(matched_01['truth']['id']==1)|(matched_01['truth']['id']==4)]
            truth_tops_2_01 = matched_01['truth']['truth'][(matched_01['truth']['id']==2)|(matched_01['truth']['id']==5)]
            truth_tops_3_01 = matched_01['truth']['truth'][(matched_01['truth']['id']==3)|(matched_01['truth']['id']==6)|(matched_01['truth']['id']==7)]

            reco_tops_1_01 = matched_01['reco'][(matched_01['truth']['id']==1)|(matched_01['truth']['id']==4)]
            reco_tops_2_01 = matched_01['reco'][(matched_01['truth']['id']==2)|(matched_01['truth']['id']==5)]
            reco_tops_3_01 = matched_01['reco'][(matched_01['truth']['id']==3)|(matched_01['truth']['id']==6)|(matched_01['truth']['id']==7)]

            matched_03 = matched_data[(lepton_indices[:,0]==0)&(lepton_indices[:,1]==3)]
            truth_tops_1_03 = matched_03['truth']['truth'][(matched_03['truth']['id']==1)|(matched_03['truth']['id']==4)]
            truth_tops_3_03 = matched_03['truth']['truth'][(matched_03['truth']['id']==2)|(matched_03['truth']['id']==5)|(matched_03['truth']['id']==6)]
            truth_tops_2_03 = matched_03['truth']['truth'][(matched_03['truth']['id']==3)|(matched_03['truth']['id']==7)]

            reco_tops_1_03 = matched_03['reco'][(matched_03['truth']['id']==1)|(matched_03['truth']['id']==4)]
            reco_tops_3_03 = matched_03['reco'][(matched_03['truth']['id']==2)|(matched_03['truth']['id']==5)|(matched_03['truth']['id']==6)]
            reco_tops_2_03 = matched_03['reco'][(matched_03['truth']['id']==3)|(matched_03['truth']['id']==7)]

            matched_23 = matched_data[(lepton_indices[:,0]==2)&(lepton_indices[:,1]==3)]
            truth_tops_3_23 = matched_23['truth']['truth'][(matched_23['truth']['id']==1)|(matched_23['truth']['id']==4)|(matched_23['truth']['id']==5)]
            truth_tops_1_23 = matched_23['truth']['truth'][(matched_23['truth']['id']==2)|(matched_23['truth']['id']==6)]
            truth_tops_2_23 = matched_23['truth']['truth'][(matched_23['truth']['id']==3)|(matched_23['truth']['id']==7)]

            reco_tops_3_23 = matched_23['reco'][(matched_23['truth']['id']==1)|(matched_23['truth']['id']==4)|(matched_23['truth']['id']==5)]
            reco_tops_1_23 = matched_23['reco'][(matched_23['truth']['id']==2)|(matched_23['truth']['id']==6)]
            reco_tops_2_23 = matched_23['reco'][(matched_23['truth']['id']==3)|(matched_23['truth']['id']==7)]

            truth_tops_1 = ak.concatenate((truth_tops_1_01, truth_tops_1_03, truth_tops_1_23))
            truth_tops_2 = ak.concatenate((truth_tops_2_01, truth_tops_2_03, truth_tops_2_23))
            truth_tops_3 = ak.concatenate((truth_tops_3_01, truth_tops_3_03, truth_tops_3_23))

            reco_tops_1 = ak.concatenate((reco_tops_1_01, reco_tops_1_03, reco_tops_1_23))
            reco_tops_2 = ak.concatenate((reco_tops_2_01, reco_tops_2_03, reco_tops_2_23))
            reco_tops_3 = ak.concatenate((reco_tops_3_01, reco_tops_3_03, reco_tops_3_23))

            truth_tops_1 = ak.mask(truth_tops_1, ak.num(truth_tops_1)==2)
            truth_tops_2 = ak.mask(truth_tops_2, ak.num(truth_tops_2)==2)
            truth_tops_3 = ak.mask(truth_tops_3, ak.num(truth_tops_3)==3)

            reco_tops_1 = ak.mask(reco_tops_1, ak.num(reco_tops_1)==2)
            reco_tops_2 = ak.mask(reco_tops_2, ak.num(reco_tops_2)==2)
            reco_tops_3 = ak.mask(reco_tops_3, ak.num(reco_tops_3)==3)

        truth_top_1 = ak.unflatten(ak.sum(truth_tops_1, axis=1), 1)
        truth_top_2 = ak.unflatten(ak.sum(truth_tops_2, axis=1), 1)
        truth_top_3 = ak.unflatten(ak.sum(truth_tops_3, axis=1), 1)

        reco_top_1 = ak.unflatten(ak.sum(reco_tops_1, axis=1), 1)
        reco_top_2 = ak.unflatten(ak.sum(reco_tops_2, axis=1), 1)
        reco_top_3 = ak.unflatten(ak.sum(reco_tops_3, axis=1), 1)

        all_truth_tops = ak.drop_none(ak.concatenate((truth_top_1,truth_top_2,truth_top_3), axis=1))
        all_reco_tops = ak.drop_none(ak.concatenate((reco_top_1,reco_top_2,reco_top_3), axis=1))

        sm_mask, dm_mask, fm_mask = (ak.num(all_truth_tops)==1), (ak.num(all_truth_tops)==2), (ak.num(all_truth_tops)==3)

        sm_truth_tops, sm_reco_tops = all_truth_tops[sm_mask], all_reco_tops[sm_mask]
        dm_truth_tops, dm_reco_tops = all_truth_tops[dm_mask], all_reco_tops[dm_mask]
        fm_truth_tops, fm_reco_tops = all_truth_tops[fm_mask], all_reco_tops[fm_mask]

        return [sm_truth_tops, sm_reco_tops], [dm_truth_tops, dm_reco_tops], [fm_truth_tops, fm_reco_tops]

def truth_matching(truth, truth_id, reco_jets, reco_leptons):
    id = np.tile(np.arange(1,len(truth[0])+1), (len(truth),1))
    print(len(truth[0]))

    indexed_truth = ak.zip({'truth': truth, 'id': id})
    
    truth_jets = indexed_truth[truth_id < 10]
    truth_leptons = indexed_truth[(truth_id > 10)&(truth_id<20)]

    jets_3d = ak.cartesian({'reco':reco_jets, 'truth': truth_jets}, axis=1, nested=True)

    deltars = jets_3d['reco'].deltaR(jets_3d['truth']['truth'])
    min_deltars = ak.min(deltars,axis=2)

    matched_jets = ak.flatten(jets_3d[(deltars==min_deltars)&(min_deltars<0.4)], axis=2)


    leptons_3d = ak.cartesian({'truth': truth_leptons, 'reco':reco_leptons}, axis=1, nested=True)

    deltars = leptons_3d['reco'].deltaR(leptons_3d['truth']['truth'])
    min_deltars = ak.min(deltars,axis=2)

    matched_leptons = ak.flatten(leptons_3d[(deltars==min_deltars)&(min_deltars<0.1)], axis=2)

    matched = ak.concatenate((matched_jets, matched_leptons), axis=1)

    matched_pairs, repeat_pairs, matched_bools = duplicate_cleaner(matched)

    matched_events = matched_pairs
    
    return matched_events, matched_bools

def top_identifier_neutrino(matched_data, neutrinos, process, lepton_channel, lepton_indices=None):
    """
    Takes the matched, duplicate free data and trys to reconstruct any tops it can
    """

    if process=="4t":

        matched_0 = matched_data[(lepton_indices[:,0]==0)]
        neutrinos_0 = neutrinos[(lepton_indices[:,0]==0)]
        truth_tops_1_0 = matched_0['truth']['truth'][(matched_0['truth']['id']==1)|(matched_0['truth']['id']==5)]
        truth_tops_2_0 = matched_0['truth']['truth'][(matched_0['truth']['id']==2)|(matched_0['truth']['id']==6)|(matched_0['truth']['id']==7)]
        truth_tops_3_0 = matched_0['truth']['truth'][(matched_0['truth']['id']==3)|(matched_0['truth']['id']==8)|(matched_0['truth']['id']==9)]
        truth_tops_4_0 = matched_0['truth']['truth'][(matched_0['truth']['id']==4)|(matched_0['truth']['id']==10)|(matched_0['truth']['id']==11)]

        reco_tops_1_0 = matched_0['reco'][(matched_0['truth']['id']==1)|(matched_0['truth']['id']==5)]
        reco_tops_2_0 = matched_0['reco'][(matched_0['truth']['id']==2)|(matched_0['truth']['id']==6)|(matched_0['truth']['id']==7)]
        reco_tops_3_0 = matched_0['reco'][(matched_0['truth']['id']==3)|(matched_0['truth']['id']==8)|(matched_0['truth']['id']==9)]
        reco_tops_4_0 = matched_0['reco'][(matched_0['truth']['id']==4)|(matched_0['truth']['id']==10)|(matched_0['truth']['id']==11)]

        matched_2 = matched_data[(lepton_indices[:,0]==2)]
        neutrinos_2 = neutrinos[(lepton_indices[:,0]==2)]
        truth_tops_2_2 = matched_2['truth']['truth'][(matched_2['truth']['id']==1)|(matched_2['truth']['id']==5)|(matched_2['truth']['id']==6)]
        truth_tops_1_2 = matched_2['truth']['truth'][(matched_2['truth']['id']==2)|(matched_2['truth']['id']==7)]
        truth_tops_3_2 = matched_2['truth']['truth'][(matched_2['truth']['id']==3)|(matched_2['truth']['id']==8)|(matched_2['truth']['id']==9)]
        truth_tops_4_2 = matched_2['truth']['truth'][(matched_2['truth']['id']==4)|(matched_2['truth']['id']==10)|(matched_2['truth']['id']==11)]

        reco_tops_2_2 = matched_2['reco'][(matched_2['truth']['id']==1)|(matched_2['truth']['id']==5)|(matched_2['truth']['id']==6)]
        reco_tops_1_2 = matched_2['reco'][(matched_2['truth']['id']==2)|(matched_2['truth']['id']==7)]
        reco_tops_3_2 = matched_2['reco'][(matched_2['truth']['id']==3)|(matched_2['truth']['id']==8)|(matched_2['truth']['id']==9)]
        reco_tops_4_2 = matched_2['reco'][(matched_2['truth']['id']==4)|(matched_2['truth']['id']==10)|(matched_2['truth']['id']==11)]

        matched_4 = matched_data[(lepton_indices[:,0]==4)]
        neutrinos_4 = neutrinos[(lepton_indices[:,0]==4)]
        truth_tops_2_4 = matched_4['truth']['truth'][(matched_4['truth']['id']==1)|(matched_4['truth']['id']==5)|(matched_4['truth']['id']==6)]
        truth_tops_3_4 = matched_4['truth']['truth'][(matched_4['truth']['id']==2)|(matched_4['truth']['id']==7)|(matched_4['truth']['id']==8)]
        truth_tops_1_4 = matched_4['truth']['truth'][(matched_4['truth']['id']==3)|(matched_4['truth']['id']==9)]
        truth_tops_4_4 = matched_4['truth']['truth'][(matched_4['truth']['id']==4)|(matched_4['truth']['id']==10)|(matched_4['truth']['id']==11)]

        reco_tops_2_4 = matched_4['reco'][(matched_4['truth']['id']==1)|(matched_4['truth']['id']==5)|(matched_4['truth']['id']==6)]
        reco_tops_3_4 = matched_4['reco'][(matched_4['truth']['id']==2)|(matched_4['truth']['id']==7)|(matched_4['truth']['id']==8)]
        reco_tops_1_4 = matched_4['reco'][(matched_4['truth']['id']==3)|(matched_4['truth']['id']==9)]
        reco_tops_4_4 = matched_4['reco'][(matched_4['truth']['id']==4)|(matched_4['truth']['id']==10)|(matched_4['truth']['id']==11)]

        matched_6 = matched_data[(lepton_indices[:,0]==6)]
        neutrinos_6 = neutrinos[(lepton_indices[:,0]==6)]
        truth_tops_2_6 = matched_6['truth']['truth'][(matched_6['truth']['id']==1)|(matched_6['truth']['id']==5)|(matched_6['truth']['id']==6)]
        truth_tops_3_6 = matched_6['truth']['truth'][(matched_6['truth']['id']==2)|(matched_6['truth']['id']==7)|(matched_6['truth']['id']==8)]
        truth_tops_4_6 = matched_6['truth']['truth'][(matched_6['truth']['id']==3)|(matched_6['truth']['id']==9)|(matched_6['truth']['id']==10)]
        truth_tops_1_6 = matched_6['truth']['truth'][(matched_6['truth']['id']==4)|(matched_6['truth']['id']==11)]

        reco_tops_2_6 = matched_6['reco'][(matched_6['truth']['id']==1)|(matched_6['truth']['id']==5)|(matched_6['truth']['id']==6)]
        reco_tops_3_6 = matched_6['reco'][(matched_6['truth']['id']==2)|(matched_6['truth']['id']==7)|(matched_6['truth']['id']==8)]
        reco_tops_4_6 = matched_6['reco'][(matched_6['truth']['id']==3)|(matched_6['truth']['id']==9)|(matched_6['truth']['id']==10)]
        reco_tops_1_6 = matched_6['reco'][(matched_6['truth']['id']==4)|(matched_6['truth']['id']==11)]

        truth_tops_1 = ak.concatenate((truth_tops_1_0, truth_tops_1_2, truth_tops_1_4, truth_tops_1_6))
        truth_tops_2 = ak.concatenate((truth_tops_2_0, truth_tops_2_2, truth_tops_2_4, truth_tops_2_6))
        truth_tops_3 = ak.concatenate((truth_tops_3_0, truth_tops_3_2, truth_tops_3_4, truth_tops_3_6))
        truth_tops_4 = ak.concatenate((truth_tops_4_0, truth_tops_4_2, truth_tops_4_4, truth_tops_4_6))

        reco_tops_1 = ak.concatenate((reco_tops_1_0, reco_tops_1_2, reco_tops_1_4, reco_tops_1_6))
        reco_tops_2 = ak.concatenate((reco_tops_2_0, reco_tops_2_2, reco_tops_2_4, reco_tops_2_6))
        reco_tops_3 = ak.concatenate((reco_tops_3_0, reco_tops_3_2, reco_tops_3_4, reco_tops_3_6))
        reco_tops_4 = ak.concatenate((reco_tops_4_0, reco_tops_4_2, reco_tops_4_4, reco_tops_4_6))

        top_neutrinos = ak.concatenate((neutrinos_0, neutrinos_2, neutrinos_4, neutrinos_6))

        truth_tops_1 = ak.concatenate((truth_tops_1,top_neutrinos), axis=1)

        truth_tops_1 = ak.mask(truth_tops_1, ak.num(truth_tops_1)==3)
        truth_tops_2 = ak.mask(truth_tops_2, ak.num(truth_tops_2)==3)
        truth_tops_3 = ak.mask(truth_tops_3, ak.num(truth_tops_3)==3)
        truth_tops_4 = ak.mask(truth_tops_4, ak.num(truth_tops_4)==3)

        reco_tops_1 = ak.concatenate((reco_tops_1,top_neutrinos), axis=1)

        reco_tops_1 = ak.mask(reco_tops_1, ak.num(reco_tops_1)==3)
        reco_tops_2 = ak.mask(reco_tops_2, ak.num(reco_tops_2)==3)
        reco_tops_3 = ak.mask(reco_tops_3, ak.num(reco_tops_3)==3)
        reco_tops_4 = ak.mask(reco_tops_4, ak.num(reco_tops_4)==3)

        truth_top_1 = ak.unflatten(ak.sum(truth_tops_1, axis=1), 1)
        truth_top_2 = ak.unflatten(ak.sum(truth_tops_2, axis=1), 1)
        truth_top_3 = ak.unflatten(ak.sum(truth_tops_3, axis=1), 1)
        truth_top_4 = ak.unflatten(ak.sum(truth_tops_4, axis=1), 1)

        reco_top_1 = ak.unflatten(ak.sum(reco_tops_1, axis=1), 1)
        reco_top_2 = ak.unflatten(ak.sum(reco_tops_2, axis=1), 1)
        reco_top_3 = ak.unflatten(ak.sum(reco_tops_3, axis=1), 1)
        reco_top_4 = ak.unflatten(ak.sum(reco_tops_4, axis=1), 1)

        all_truth_tops = ak.drop_none(ak.concatenate((truth_top_1,truth_top_2,truth_top_3,truth_top_4), axis=1))
        all_reco_tops = ak.drop_none(ak.concatenate((reco_top_1,reco_top_2,reco_top_3,reco_top_4), axis=1))

        sm_mask, dm_mask, tm_mask, fm_mask = (ak.num(all_truth_tops)==1), (ak.num(all_truth_tops)==2), (ak.num(all_truth_tops)==3), (ak.num(all_truth_tops)==4)

        sm_truth_tops, sm_reco_tops = all_truth_tops[sm_mask], all_reco_tops[sm_mask]
        dm_truth_tops, dm_reco_tops = all_truth_tops[dm_mask], all_reco_tops[dm_mask]
        tm_truth_tops, tm_reco_tops = all_truth_tops[tm_mask], all_reco_tops[tm_mask]
        fm_truth_tops, fm_reco_tops = all_truth_tops[fm_mask], all_reco_tops[fm_mask]

        return [sm_truth_tops, sm_reco_tops], [dm_truth_tops, dm_reco_tops], [tm_truth_tops, tm_reco_tops], [fm_truth_tops, fm_reco_tops]
    elif (process=="3tj")|(process=="3tW"):

        matched_0 = matched_data[(lepton_indices[:,0]==0)]
        neutrinos_0 = neutrinos[(lepton_indices[:,0]==0)]
        truth_tops_1_0 = matched_0['truth']['truth'][(matched_0['truth']['id']==1)|(matched_0['truth']['id']==4)]
        truth_tops_2_0 = matched_0['truth']['truth'][(matched_0['truth']['id']==2)|(matched_0['truth']['id']==5)|(matched_0['truth']['id']==6)]
        truth_tops_3_0 = matched_0['truth']['truth'][(matched_0['truth']['id']==3)|(matched_0['truth']['id']==7)|(matched_0['truth']['id']==8)]

        reco_tops_1_0 = matched_0['reco'][(matched_0['truth']['id']==1)|(matched_0['truth']['id']==4)]
        reco_tops_2_0 = matched_0['reco'][(matched_0['truth']['id']==2)|(matched_0['truth']['id']==5)|(matched_0['truth']['id']==6)]
        reco_tops_3_0 = matched_0['reco'][(matched_0['truth']['id']==3)|(matched_0['truth']['id']==7)|(matched_0['truth']['id']==8)]

        matched_2 = matched_data[(lepton_indices[:,0]==2)]
        neutrinos_2 = neutrinos[(lepton_indices[:,0]==2)]
        truth_tops_2_2 = matched_2['truth']['truth'][(matched_2['truth']['id']==1)|(matched_2['truth']['id']==4)|(matched_2['truth']['id']==5)]
        truth_tops_1_2 = matched_2['truth']['truth'][(matched_2['truth']['id']==2)|(matched_2['truth']['id']==6)]
        truth_tops_3_2 = matched_2['truth']['truth'][(matched_2['truth']['id']==3)|(matched_2['truth']['id']==7)|(matched_2['truth']['id']==8)]

        reco_tops_2_2 = matched_2['reco'][(matched_2['truth']['id']==1)|(matched_2['truth']['id']==4)|(matched_2['truth']['id']==5)]
        reco_tops_1_2 = matched_2['reco'][(matched_2['truth']['id']==2)|(matched_2['truth']['id']==6)]
        reco_tops_3_2 = matched_2['reco'][(matched_2['truth']['id']==3)|(matched_2['truth']['id']==7)|(matched_2['truth']['id']==8)]

        matched_4 = matched_data[(lepton_indices[:,0]==4)]
        neutrinos_4 = neutrinos[(lepton_indices[:,0]==4)]
        truth_tops_2_4 = matched_4['truth']['truth'][(matched_4['truth']['id']==1)|(matched_4['truth']['id']==4)|(matched_4['truth']['id']==5)]
        truth_tops_3_4 = matched_4['truth']['truth'][(matched_4['truth']['id']==2)|(matched_4['truth']['id']==6)|(matched_4['truth']['id']==7)]
        truth_tops_1_4 = matched_4['truth']['truth'][(matched_4['truth']['id']==3)|(matched_4['truth']['id']==8)]

        reco_tops_2_4 = matched_4['reco'][(matched_4['truth']['id']==1)|(matched_4['truth']['id']==4)|(matched_4['truth']['id']==5)]
        reco_tops_3_4 = matched_4['reco'][(matched_4['truth']['id']==2)|(matched_4['truth']['id']==6)|(matched_4['truth']['id']==7)]
        reco_tops_1_4 = matched_4['reco'][(matched_4['truth']['id']==3)|(matched_4['truth']['id']==8)]

        truth_tops_1 = ak.concatenate((truth_tops_1_0, truth_tops_1_2, truth_tops_1_4))
        truth_tops_2 = ak.concatenate((truth_tops_2_0, truth_tops_2_2, truth_tops_2_4))
        truth_tops_3 = ak.concatenate((truth_tops_3_0, truth_tops_3_2, truth_tops_3_4))

        reco_tops_1 = ak.concatenate((reco_tops_1_0, reco_tops_1_2, reco_tops_1_4))
        reco_tops_2 = ak.concatenate((reco_tops_2_0, reco_tops_2_2, reco_tops_2_4))
        reco_tops_3 = ak.concatenate((reco_tops_3_0, reco_tops_3_2, reco_tops_3_4))

        top_neutrinos = ak.concatenate((neutrinos_0, neutrinos_2, neutrinos_4))

        truth_tops_1 = ak.concatenate((truth_tops_1,top_neutrinos), axis=1)

        truth_tops_1 = ak.mask(truth_tops_1, ak.num(truth_tops_1)==3)
        truth_tops_2 = ak.mask(truth_tops_2, ak.num(truth_tops_2)==3)
        truth_tops_3 = ak.mask(truth_tops_3, ak.num(truth_tops_3)==3)

        reco_tops_1 = ak.concatenate((reco_tops_1,top_neutrinos), axis=1)

        reco_tops_1 = ak.mask(reco_tops_1, ak.num(reco_tops_1)==3)
        reco_tops_2 = ak.mask(reco_tops_2, ak.num(reco_tops_2)==3)
        reco_tops_3 = ak.mask(reco_tops_3, ak.num(reco_tops_3)==3)

        truth_top_1 = ak.unflatten(ak.sum(truth_tops_1, axis=1), 1)
        truth_top_2 = ak.unflatten(ak.sum(truth_tops_2, axis=1), 1)
        truth_top_3 = ak.unflatten(ak.sum(truth_tops_3, axis=1), 1)

        reco_top_1 = ak.unflatten(ak.sum(reco_tops_1, axis=1), 1)
        reco_top_2 = ak.unflatten(ak.sum(reco_tops_2, axis=1), 1)
        reco_top_3 = ak.unflatten(ak.sum(reco_tops_3, axis=1), 1)

        all_truth_tops = ak.drop_none(ak.concatenate((truth_top_1,truth_top_2,truth_top_3), axis=1))
        all_reco_tops = ak.drop_none(ak.concatenate((reco_top_1,reco_top_2,reco_top_3), axis=1))

        sm_mask, dm_mask, fm_mask = (ak.num(all_truth_tops)==1), (ak.num(all_truth_tops)==2), (ak.num(all_truth_tops)==3)

        sm_truth_tops, sm_reco_tops = all_truth_tops[sm_mask], all_reco_tops[sm_mask]
        dm_truth_tops, dm_reco_tops = all_truth_tops[dm_mask], all_reco_tops[dm_mask]
        fm_truth_tops, fm_reco_tops = all_truth_tops[fm_mask], all_reco_tops[fm_mask]

        return [sm_truth_tops, sm_reco_tops], [dm_truth_tops, dm_reco_tops], [fm_truth_tops, fm_reco_tops]

def data_generator(input_file, lepton_channel, top_channel, reco_only=False):
    
    tree = uproot.open(f"{input_file}")
    tree_truth = tree['Truth;1']
    tree_reco = tree['Reco;1']
    
    events_truth = tree_truth.arrays(['b_id', 'b_pt', 'b_eta', 'b_phi', 'b_e', 'b_mass', 'W_decay_id', 'W_decay_pt', 'W_decay_eta', 'W_decay_phi', 'W_decay_e', 'W_decay_mass'])
    events_reco = tree_reco.arrays(['jet_pt', 'jet_eta', 'jet_phi', 'jet_mass', 'jet_btag', 'el_pt', 'el_eta', 'el_phi', 'el_charge', 'mu_pt', 'mu_eta', 'mu_phi', 'mu_charge'])
    events_reco['el_mass'] = 0.511e-3 * np.ones_like(events_reco['el_pt'])
    events_reco['mu_mass'] = 0.1057 * np.ones_like(events_reco['mu_pt'])
    events_reco['jet_e'] = np.sqrt(events_reco['jet_mass']**2+(events_reco['jet_pt']**2)*(np.cosh(events_reco['jet_eta']))**2)
    events_reco['el_e'] = np.sqrt(events_reco['el_mass']**2+(events_reco['el_pt']**2)*(np.cosh(events_reco['el_eta']))**2)
    events_reco['mu_e'] = np.sqrt(events_reco['mu_mass']**2+(events_reco['mu_pt']**2)*(np.cosh(events_reco['mu_eta']))**2)

    jet_pt = events_reco['jet_pt'][(events_reco['jet_eta']<2.5) & (events_reco['jet_eta']>-2.5)]
    jet_eta = events_reco['jet_eta'][(events_reco['jet_eta']<2.5) & (events_reco['jet_eta']>-2.5)]
    jet_phi = events_reco['jet_phi'][(events_reco['jet_eta']<2.5) & (events_reco['jet_eta']>-2.5)]
    jet_mass = events_reco['jet_mass'][(events_reco['jet_eta']<2.5) & (events_reco['jet_eta']>-2.5)]
    jet_e = events_reco['jet_e'][(events_reco['jet_eta']<2.5) & (events_reco['jet_eta']>-2.5)]
    
    electron_pt = events_reco['el_pt'][(events_reco['el_pt'] > 15) & (events_reco['el_eta'] < 2.47) & (events_reco['el_eta'] > -2.47)]
    electron_eta = events_reco['el_eta'][(events_reco['el_pt'] > 15) & (events_reco['el_eta'] < 2.47) & (events_reco['el_eta'] > -2.47)]
    electron_phi = events_reco['el_phi'][(events_reco['el_pt'] > 15) & (events_reco['el_eta'] < 2.47) & (events_reco['el_eta'] > -2.47)]
    electron_mass = events_reco['el_mass'][(events_reco['el_pt'] > 15) & (events_reco['el_eta'] < 2.47) & (events_reco['el_eta'] > -2.47)]
    electron_e = events_reco['el_e'][(events_reco['el_pt'] > 15) & (events_reco['el_eta'] < 2.47) & (events_reco['el_eta'] > -2.47)]
    
    muon_pt = events_reco['mu_pt'][(events_reco['mu_pt']>15) & (events_reco['mu_eta']<2.5) & (events_reco['mu_eta']>-2.5)]
    muon_eta = events_reco['mu_eta'][(events_reco['mu_pt']>15) & (events_reco['mu_eta']<2.5) & (events_reco['mu_eta']>-2.5)]
    muon_phi = events_reco['mu_phi'][(events_reco['mu_pt']>15) & (events_reco['mu_eta']<2.5) & (events_reco['mu_eta']>-2.5)]
    muon_mass = events_reco['mu_mass'][(events_reco['mu_pt']>15) & (events_reco['mu_eta']<2.5) & (events_reco['mu_eta']>-2.5)]
    muon_e = events_reco['mu_e'][(events_reco['mu_pt']>15) & (events_reco['mu_eta']<2.5) & (events_reco['mu_eta']>-2.5)]
    
    lepton_pt = ak.concatenate((muon_pt,electron_pt), axis=1)
    lepton_eta = ak.concatenate((muon_eta,electron_eta), axis=1)
    lepton_phi = ak.concatenate((muon_phi,electron_phi), axis=1)
    lepton_mass = ak.concatenate((muon_mass,electron_mass), axis=1)
    lepton_e = ak.concatenate((muon_e,electron_e), axis=1)
    
    reco_pt = ak.concatenate((jet_pt,lepton_pt), axis=1)
    reco_eta = ak.concatenate((jet_eta,lepton_eta), axis=1)
    reco_phi = ak.concatenate((jet_phi,lepton_phi), axis=1)
    reco_mass = ak.concatenate((jet_mass,lepton_mass), axis=1)
    reco_e = ak.concatenate((jet_e,lepton_e), axis=1)
    
    b_pt = events_truth['b_pt']
    b_eta = events_truth['b_eta']
    b_phi = events_truth['b_phi']
    b_mass = events_truth['b_mass']
    b_e = events_truth['b_e']
    b_id = events_truth['b_id']
    
    w_decay_pt = events_truth['W_decay_pt']
    w_decay_eta = events_truth['W_decay_eta']
    w_decay_phi = events_truth['W_decay_phi']
    w_decay_mass = events_truth['W_decay_mass']
    w_decay_e = events_truth['W_decay_e']
    w_decay_id = events_truth['W_decay_id']
    
    truth_pt = ak.concatenate((b_pt,w_decay_pt), axis=1)
    truth_eta = ak.concatenate((b_eta,w_decay_eta), axis=1)
    truth_phi = ak.concatenate((b_phi,w_decay_phi), axis=1)
    truth_mass = ak.concatenate((b_mass,w_decay_mass), axis=1)
    truth_e = ak.concatenate((b_e,w_decay_e), axis=1)
    truth_id = ak.concatenate((b_id,w_decay_id), axis=1)
    
    reco = vector.zip({'pt':reco_pt,'eta':reco_eta,'phi':reco_phi,'energy':reco_e})
    truth = vector.zip({'pt':truth_pt,'eta':truth_eta,'phi':truth_phi,'energy':truth_e})
    
    reco_leptons = vector.zip({'pt':lepton_pt,'eta':lepton_eta,'phi':lepton_phi,'energy':lepton_e})
    reco_jets = vector.zip({'pt':jet_pt,'eta':jet_eta,'phi':jet_phi,'energy':jet_e})
    truth = truth[(ak.num(reco)!=0)]
    reco = reco[(ak.num(reco)!=0)]
    reco_leptons = reco_leptons[(ak.num(reco)!=0)]
    reco_jets = reco_jets[(ak.num(reco)!=0)]

    lepton_count = ak.num(events_truth['W_decay_id'][(abs(events_truth['W_decay_id'])>10)&(abs(events_truth['W_decay_id'])<19)])
    mask = lepton_count == int(lepton_channel[0])
    events_truth = events_truth[mask]
    #truth, truth_id = truth[mask], truth_id[mask]

    reco, reco_leptons, reco_jets = reco[mask], reco_leptons[mask], reco_jets[mask]

    if reco_only:
        return reco, reco_leptons, reco_jets, events_reco

    matched_events, matched_bool = truth_matching(truth, truth_id, reco_jets, reco_leptons)
    matched = ak.zip({'reco': matched_events['reco'], 'truth': matched_events['truth']})
    lepton_indices = ak.sort(ak.argsort(abs(events_truth['W_decay_id']), axis = 1, ascending=False)[:,:int(lepton_channel[0])], axis=1)

    if lepton_channel == "1L":
        tree_truth_neutrino = tree['Truth;Neutrino']
        events_truth_neutrino = tree_truth_neutrino.arrays(['W_decay_id', 'W_decay_pt', 'W_decay_eta', 'W_decay_phi', 'W_decay_e', 'W_decay_mass'])
        truth_neutrino_pt = events_truth_neutrino['W_decay_pt']
        truth_neutrino_eta = events_truth_neutrino['W_decay_eta']
        truth_neutrino_phi = events_truth_neutrino['W_decay_phi']
        truth_neutrino_e = events_truth_neutrino['W_decay_e']
        
        tree_reco_neutrino = tree['Reco;Neutrino']
        events_reco_neutrino = tree_reco_neutrino.arrays(['met_met', 'met_eta', 'met_phi', 'met_e'])
        reco_neutrino_pt = events_reco_neutrino['met_met']
        reco_neutrino_eta = events_reco_neutrino['met_eta']
        reco_neutrino_phi = events_reco_neutrino['met_phi']
        reco_neutrino_e = events_reco_neutrino['met_e']

        truth_neutrinos = vector.zip({'pt':truth_neutrino_pt,'eta':truth_neutrino_eta,'phi':truth_neutrino_phi,'energy':truth_neutrino_e})
        print(len(truth_neutrinos))
        reco_neutrinos = vector.zip({'pt':reco_neutrino_pt,'eta':reco_neutrino_eta,'phi':reco_neutrino_phi,'energy':reco_neutrino_e})
        array = top_identifier_neutrino(matched, truth_neutrinos, top_channel, lepton_channel, lepton_indices=lepton_indices)
        print(len(array[0][0]),len(array[1][0]),len(array[2][0]))
    else:
        
        array = top_identifier(matched,top_channel,lepton_channel, lepton_indices=lepton_indices)
    print(array)
    truth_tops, reco_tops = array[2]

    print(len(truth_tops))

    return truth_tops, reco_tops, reco

def centrality(particle, reference_1, reference_2):
    return np.abs(particle.rapidity - 0.5*(reference_1.rapidity + reference_2.rapidity))/np.abs(reference_1.rapidity-reference_2.rapidity)

def sphericity(reco_tops):
    sum_pt = np.sum(reco_tops.pt, axis=1)
    S_xy = np.zeros((len(reco_tops),2,2))

    S_xy[:,0,0] = np.sum(reco_tops.px**2/reco_tops.pt, axis=1)
    S_xy[:,0,1] = np.sum(reco_tops.px*reco_tops.py/reco_tops.pt, axis=1)
    S_xy[:,1,0] = S_xy[:,0,1]
    S_xy[:,1,1] = np.sum(reco_tops.py**2/reco_tops.pt, axis=1)

    S_xy = S_xy/sum_pt[:,np.newaxis,np.newaxis]

    eigenvalues = np.linalg.eigvalsh(S_xy)
    sphericity = 2*np.min(eigenvalues, axis=1)/np.sum(eigenvalues, axis=1)

    return sphericity

def sorted_tops(reco_tops):
    sorted_tops_pt = reco_tops[ak.argsort(reco_tops.pt)]
    highest_pt_top = sorted_tops_pt[:,-1]
    second_highest_pt_top = sorted_tops_pt[:,-2]
    lowest_pt_top = sorted_tops_pt[:,0]
    print(np.shape(highest_pt_top.pt))
    sorted_pt_tops = np.vstack((highest_pt_top.pt,second_highest_pt_top.pt,lowest_pt_top.pt))

    print(np.shape(sorted_pt_tops))

    sorted_tops_eta = reco_tops[ak.argsort(reco_tops.eta)]
    highest_eta_top = sorted_tops_eta[:,-1]
    second_highest_eta_top = sorted_tops_eta[:,-2]
    lowest_eta_top = sorted_tops_eta[:,0]

    sorted_eta_tops = np.vstack((highest_eta_top.eta,second_highest_eta_top.eta,lowest_eta_top.eta))
    sorted_tops_phi = reco_tops[ak.argsort(reco_tops.phi)]
    highest_phi_top = sorted_tops_phi[:,-1]
    second_highest_phi_top = sorted_tops_phi[:,-2]
    lowest_phi_top = sorted_tops_phi[:,0]

    sorted_phi_tops = np.vstack((highest_phi_top.phi,second_highest_phi_top.phi,lowest_phi_top.phi))

    sorted_tops_E = reco_tops[ak.argsort(reco_tops.E)]
    highest_E_top = sorted_tops_E[:,-1]
    second_highest_E_top = sorted_tops_E[:,-2]
    lowest_E_top = sorted_tops_E[:,0]

    sorted_E_tops = np.vstack((highest_E_top.E,second_highest_E_top.E,lowest_E_top.E))

    sorted_tops = np.vstack((sorted_pt_tops,sorted_eta_tops,sorted_phi_tops,sorted_E_tops))

    print(np.shape(sorted_tops))

    return sorted_tops_pt, sorted_tops

def average_variables(reco_tops):
    duos = ak.combinations(reco_tops, 2, axis=1)

    duo_masses = np.mean((duos["0"]+duos["1"]).m, axis=1)
    deltaeta = np.mean(duos["0"].deltaeta(duos["1"]), axis=1)
    deltaphi = np.mean(duos["0"].deltaphi(duos["1"]), axis=1)
    deltaR = np.mean(duos["0"].deltaR(duos["1"]), axis=1)
    deltapt = np.mean((duos["0"].pt-duos["1"].pt), axis=1)

    averages = np.vstack((duo_masses, deltaeta, deltaphi, deltaR, deltapt))

    print(np.shape(averages))

    return averages

def total_tops_variables(reco_tops):
    total_tops = np.sum(reco_tops,axis=1)
    H_t = np.sum(reco_tops.pt,axis=1)
    invariant_mass = np.sum(reco_tops.m,axis=1)
    sum_inv_mass = total_tops.m
    sum_pt = total_tops.pt

    total_variables = np.vstack((H_t, invariant_mass, sum_inv_mass, sum_pt))
    print(np.shape(total_variables))
    return total_variables

def combined_array(centrality, sphericity, sorted_tops, average_variables, total_variables):
    combined_data_trans = np.vstack((centrality, sphericity, sorted_tops, average_variables, total_variables))
    combined_data = np.transpose(combined_data_trans)
    return combined_data

def main(infile,outfile,channel):
    
    """
    For parsing directly from a ROOT file 
    """
    reco_tops, reco_leptons, reco_jets = data_generator(infile, "1L", channel)
    print(f"top pt data:{reco_tops.pt}")
    print(f"Tops shape:{np.shape(reco_tops)}")
    sorted_tops_pt, sorted_tops_array = sorted_tops(reco_tops)
    centrality_tops = centrality(sorted_tops_pt[:,0], sorted_tops_pt[:,1], sorted_tops_pt[:,2])
    sphericity_tops = sphericity(reco_tops)
    average_variables_tops = average_variables(reco_tops)
    total_top_variables = total_tops_variables(reco_tops)
    combined_top_variables = combined_array(centrality_tops, sphericity_tops, sorted_tops_array, average_variables_tops, total_top_variables)
    print(np.shape(combined_top_variables))

    sorted_reco_pt, sorted_reco_array = sorted_tops(reco_jets)
    centrality_reco = centrality(sorted_reco_pt[:,0], sorted_reco_pt[:,1], sorted_reco_pt[:,2])
    sphericity_reco = sphericity(reco_jets)
    average_variables_reco = average_variables(reco_jets)
    total_reco_variables = total_tops_variables(reco_jets)
    combined_reco_variables = combined_array(centrality_reco, sphericity_reco, sorted_reco_array, average_variables_reco, total_reco_variables)
    print(np.shape(combined_reco_variables))

    combined_variables_all = np.hstack((combined_top_variables, combined_reco_variables))
    print(np.shape(combined_variables_all))

    label = 0 if channel == "4t" else 1

    with h5py.File(outfile, 'w') as hf:
        hf.create_dataset('INPUTS', data=combined_reco_variables)
        hf.create_dataset('LABELS', data=[label]*len(combined_reco_variables))



if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    channel = sys.argv[3]
    main(infile, outfile, channel)