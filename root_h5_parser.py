import uproot
import numpy as np
import awkward as ak
import vector
import h5py 
import sys

def data_generator(input_file):
    
    tree = uproot.open(f"{input_file}")
    tree_reco = tree['Reco;1']
    tree_tops = tree['Top;1']

    reco_pt = tree_reco['pt']
    reco_eta = tree_reco['eta']
    reco_phi = tree_reco['phi']
    reco_energy = tree_reco['energy']

    top_pt = tree_tops['pt']
    top_eta = tree_tops['eta']
    top_phi = tree_tops['phi']
    top_energy = tree_tops['energy']
    
    reco = vector.zip({'pt':reco_pt, 'eta':reco_eta, 'phi':reco_phi, 'E':reco_energy})
    tops = vector.zip({'pt':top_pt, 'eta':top_eta, 'phi':top_phi, 'E':top_energy})

    return tops, reco

def centrality(particle, reference_1, reference_2):
    return np.abs(particle.rapidity - 0.5*(reference_1.rapidity + reference_2.rapidity))/np.abs(reference_1.rapidity-reference_2.rapidity)

def sphericity(particles):
    sum_pt = np.sum(particles.pt, axis=1)
    S_xy = np.zeros((len(particles),2,2))

    S_xy[:,0,0] = np.sum(particles.px**2/particles.pt, axis=1)
    S_xy[:,0,1] = np.sum(particles.px*particles.py/particles.pt, axis=1)
    S_xy[:,1,0] = S_xy[:,0,1]
    S_xy[:,1,1] = np.sum(particles.py**2/particles.pt, axis=1)

    S_xy = S_xy/sum_pt[:,np.newaxis,np.newaxis]

    eigenvalues = np.linalg.eigvalsh(S_xy)
    sphericity = 2*np.min(eigenvalues, axis=1)/np.sum(eigenvalues, axis=1)

    return sphericity

def sorted_variables(particles):
    sorted_tops_pt = particles[ak.argsort(particles.pt)]
    highest_pt_top = sorted_tops_pt[:,-1]
    second_highest_pt_top = sorted_tops_pt[:,-2]
    lowest_pt_top = sorted_tops_pt[:,0]

    sorted_pt_tops = np.vstack((highest_pt_top.pt,second_highest_pt_top.pt,lowest_pt_top.pt))

    sorted_tops_eta = particles[ak.argsort(particles.eta)]
    highest_eta_top = sorted_tops_eta[:,-1]
    second_highest_eta_top = sorted_tops_eta[:,-2]
    lowest_eta_top = sorted_tops_eta[:,0]

    sorted_eta_tops = np.vstack((highest_eta_top.eta,second_highest_eta_top.eta,lowest_eta_top.eta))
    sorted_tops_phi = particles[ak.argsort(particles.phi)]
    highest_phi_top = sorted_tops_phi[:,-1]
    second_highest_phi_top = sorted_tops_phi[:,-2]
    lowest_phi_top = sorted_tops_phi[:,0]

    sorted_phi_tops = np.vstack((highest_phi_top.phi,second_highest_phi_top.phi,lowest_phi_top.phi))

    sorted_tops_E = particles[ak.argsort(particles.E)]
    highest_E_top = sorted_tops_E[:,-1]
    second_highest_E_top = sorted_tops_E[:,-2]
    lowest_E_top = sorted_tops_E[:,0]

    sorted_E_tops = np.vstack((highest_E_top.E,second_highest_E_top.E,lowest_E_top.E))

    sorted_tops = np.vstack((sorted_pt_tops,sorted_eta_tops,sorted_phi_tops,sorted_E_tops))

    return sorted_tops_pt, sorted_tops

def average_variables(particles):
    duos = ak.combinations(particles, 2, axis=1)

    duo_masses = np.mean((duos["0"]+duos["1"]).m, axis=1)
    deltaeta = np.mean(duos["0"].deltaeta(duos["1"]), axis=1)
    deltaphi = np.mean(duos["0"].deltaphi(duos["1"]), axis=1)
    deltaR = np.mean(duos["0"].deltaR(duos["1"]), axis=1)
    deltapt = np.mean((duos["0"].pt-duos["1"].pt), axis=1)

    averages = np.vstack((duo_masses, deltaeta, deltaphi, deltaR, deltapt))

    return averages

def total_tops_variables(particles):
    total_tops = np.sum(particles,axis=1)
    H_t = np.sum(particles.pt,axis=1)
    invariant_mass = np.sum(particles.m,axis=1)
    sum_inv_mass = total_tops.m
    sum_pt = total_tops.pt

    total_variables = np.vstack((H_t, invariant_mass, sum_inv_mass, sum_pt))
    return total_variables

def combined_array(centrality, sphericity, sorted_tops, average_variables, total_variables):
    combined_data_trans = np.vstack((centrality, sphericity, sorted_tops, average_variables, total_variables))
    combined_data = np.transpose(combined_data_trans)
    return combined_data

def main(infile,outfile_base,decay_channel,lepton_channel):
    
    """
    For parsing directly from a ROOT file 
    """
    tops, reco = data_generator(infile)

    sorted_tops_pt, sorted_tops_array = sorted_variables(tops)
    centrality_tops = centrality(sorted_tops_pt[:,0], sorted_tops_pt[:,1], sorted_tops_pt[:,2])
    sphericity_tops = sphericity(tops)
    average_variables_tops = average_variables(tops)
    total_top_variables = total_tops_variables(tops)
    top_variables = combined_array(centrality_tops, sphericity_tops, sorted_tops_array, average_variables_tops, total_top_variables)

    sorted_reco_pt, sorted_reco_array = sorted_variables(reco)
    centrality_reco = centrality(sorted_reco_pt[:,0], sorted_reco_pt[:,1], sorted_reco_pt[:,2])
    sphericity_reco = sphericity(reco)
    average_variables_reco = average_variables(reco)
    total_reco_variables = total_tops_variables(reco)
    reco_variables = combined_array(centrality_reco, sphericity_reco, sorted_reco_array, average_variables_reco, total_reco_variables)

    combined_variables = np.hstack((top_variables, reco_variables))
    
    label = 0 if int(decay_channel[0]) == 4 else 1

    print(label)

    outfile_reco = f"{outfile_base}_reco.h5"
    outfile_tops = f"{outfile_base}_tops.h5"
    outfile_combined = f"{outfile_base}_combined.h5"

    with h5py.File(outfile_reco, 'w') as hf:
        hf.create_dataset('INPUTS', data=reco_variables)
        hf.create_dataset('LABELS', data=[label]*len(reco_variables))

    with h5py.File(outfile_tops, 'w') as hf:
        hf.create_dataset('INPUTS', data=top_variables)
        hf.create_dataset('LABELS', data=[label]*len(top_variables))
    
    with h5py.File(outfile_combined, 'w') as hf:
        hf.create_dataset('INPUTS', data=combined_variables)
        hf.create_dataset('LABELS', data=[label]*len(combined_variables))


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    decay_channel = sys.argv[3]
    lepton_channel = sys.argv[4]
    main(infile, outfile, decay_channel,lepton_channel)