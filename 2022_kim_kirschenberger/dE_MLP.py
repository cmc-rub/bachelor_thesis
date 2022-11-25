# Imports
from RIC_bond import all_grad_bond, all_bond
from RIC_angle import all_grad_angle, all_angle, gen_angle_table
from RIC_dihedral import all_grad_dihedral, all_dihedral, gen_dihedral_table
from refdata_hdf5_class import refdata

import molsys

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, value_and_grad
from jax import random
from jax.config import config
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
#config.update("jax_enable_x64", True)
import time
import os
import pickle
from datetime import datetime


# RICS
#@jit
def get_all_RICs(xyz, ric_tables):
    # broadcast to fit iteration cycles with the same RICS, before vmap?
    b = all_bond(xyz, ric_tables[0])
    a = all_angle(xyz, ric_tables[1])
    d = all_dihedral(xyz, ric_tables[2])
    return jnp.concatenate((b,a,d))

@jit
def mbatch_RICs(xyz, ric_tables):
    """vmap wrapper for calculating all RICS for a mini batch"""
    return vmap(get_all_RICs, in_axes = (0, None))(xyz, ric_tables)

# Core structure and functions
@jit
def MLP(params, inp): # note: still uses a for loop ...
    """forward"""
    curr_layer = inp
    for w, b in params[:-1]: # only till the last one, because then NO activation function is used!
        curr_layer = jax.nn.gelu(jnp.dot(w, curr_layer)+b)
        #curr_layer = jnp.tanh(jnp.dot(w, curr_layer)+b)
        #curr_layer = jnp.dot(w, curr_layer)+b --> generiert "normalverteilung"
    # output layer
    w, b = params[-1]
    out_layer = jnp.dot(w, curr_layer) + b
    return out_layer[0] # out is size 1, delta Energy


@jit
def mbatch_MLP(params, inp):
    """vmap wrapper for MLP for mini batches"""
    return vmap(MLP, in_axes = (None, 0))(params, inp)

# Loss function
@jit
def loss(params, inp, ref_value):
    """rms of the deltas --> for a single data point """
    energy = MLP(params, inp)
    delta = energy-ref_value
    rms = jnp.sqrt(jnp.mean(jnp.square(delta)))
    return rms
    
@jit
def get_delta(params, inp, ref_values):
    energy = MLP(params, inp)
    delta = energy-ref_values
    return delta

@jit
def mbatch_loss(params, inp, ref_values):
    """
    vmap wrapper for the loss function over a MINI BATCH. Takes the rms of all rms.
    @param ric: all rics for all the structures of the mini batch
    @param params: starting params of the mini batch (same for all -> vmap none)
    @param ref_values: ref forces for all of the structures
    """
    delta = vmap(get_delta, in_axes = (None, 0, 0))(params, inp, ref_values)
    total_rms = jnp.sqrt(jnp.mean(jnp.square(delta)))
    return total_rms

@jit
def grad_mbatch_loss(params, inp, ref_values):
    return grad(mbatch_loss, 0)(params, inp, ref_values)

# Optimizer
@jit
def ADAM_SD(params, inp, ref_values, t = 0, m = None, v = None, b1 = 0.9, b2 = 0.999, a = 0.001):
    """ 
    Adam Optimizer (doi: https://doi.org/10.48550/arXiv.1412.6980)
    """
    #print(t, m, v)
    # params adapted for different systems
    e = 10**(-8)

    grad = grad_mbatch_loss(params, inp, ref_values) # grad wrt params of loss
    t += 1 # m and v musst be the same shape as params ... -> zip + init the same as params!
    m = [(b1 * m_dw + (1-b1) * dw, b1 * m_db + (1-b1) * db) 
         for (dw, db), (m_dw, m_db) in zip(grad, m)] # m = b1 * m + (1-b1) *  grad
    grad2 = [(dw**2, db**2) for (dw, db) in grad] # grad2 = grad**2 , elementwise
    v = [(b2 * v_dw + (1-b2) * dw, b2 * v_db + (1-b2) * db) 
         for (dw, db), (v_dw, v_db) in zip(grad2, v)] # b2 * v + (1-b2) * (grad*grad)
    #print("v: ", v)
    _m = [(dw/(1-jnp.power(b1, t)), db/(1-jnp.power(b1, t))) for (dw, db) in m] # m/(1-jnp.power(b1, t))
    _v = [(dw/(1-jnp.power(b2, t)), db/(1-jnp.power(b2, t))) for (dw, db) in v] # v/(1-jnp.power(b2, t))
    #print("_m: ", _m)
    #print("_v: ", _v)
    params = [(w - (a * m_dw)/(jnp.sqrt(v_dw) + e), b - (a * m_db)/(jnp.sqrt(v_db) + e)) 
              for (w,b), (m_dw, m_db), (v_dw, v_db) in zip(params, _m, _v)] # params - (a * _m)/(jnp.sqrt(_v) + e)
    loss = mbatch_loss(params, inp, ref_values)
    return t, m, v, params, loss

# ADAM PARAMS
def init_zero_params(sizes):
    return [(jnp.zeros(shape = (n, m)), jnp.zeros(shape = n, )) for m, n in zip(sizes[:-1], sizes[1:])]

# params
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def dataloader(fname, n_samples):
    """
    Loads all relevant data from hdf5 file
    """
    r = refdata(fname = fname)
    all_xyz, all_id = r.get_xyz_from_sample(sample = "MD")
    _max = len(all_xyz)
    r.change_lot("TPSS")
    dft0_id = r.get_lowest_energy()# get lowest energy INDEX in dataset
    dft0_E = r.get_energies_from_id(dft0_id)
    train_DFT_energies = r.get_energies_from_id([x for x in range(n_samples)])
    train_DFT_energies -= dft0_E # Set refenergy

    r.change_lot("MOFFF")
    train_FF_energies = r.get_energies_from_id([x for x in range(n_samples)])
    ff0_E = r.get_energies_from_id(dft0_id)
    train_FF_energies -= ff0_E
    #sample_energies = sample_FF_energies # direct fit
    train_energies = train_DFT_energies - train_FF_energies # Delta learning
    train_xyz = all_xyz[:n_samples]

    # validation set
    r.change_lot("TPSS")
    val_DFT_energies = r.get_energies_from_id([x for x in range(n_samples,_max)]) # methods only produces "forward" indexing 
    val_DFT_energies -= dft0_E
    r.change_lot("MOFFF")
    val_FF_energies = r.get_energies_from_id([x for x in range(n_samples,_max)])
    val_FF_energies -= ff0_E
    #val_energies = val_FF_energies # direct fit
    val_energies = val_DFT_energies - val_FF_energies # Delta learning
    val_xyz = all_xyz[n_samples:_max]
    r.close()
    return train_xyz, train_energies, val_xyz, val_energies


def training(fname = "tmp", datafile = "trainingsdata/3000K_methanol.hdf5", 
             layer_sizes = None, a = None, n_samples = None, batch = None, 
             create_plot = None, save_plot = None, verbose = None,
             ini_params = None, par_id = None, maxiter = 0, seed = 0):
    """
    Combining all building blocks to a NN with a training loop
    @param fname:str        -> base for all files generated by the program
    @param datafile:str     -> path to file containing the trainings data
    @param layer_sizes:list -> layer geometry, list of all layer sizes including in and out
    @param a:float          -> (initial) step size for ADAM optimizer
    @param n_samples:int    -> number of traing points (val NOT effected!)
    @param batch:list(tuple)-> list! of sizes of the individual batches in a tuple with their number of epochs
    @param create_plot:bool -> generate a loss - epoch plot
    @param save_plot:bool   -> save the generated plot (wont save plot if none is generated)
    @param verbose:int      -> level of verbosity (0 -> nothing printed; 1 -> basic information about progress; 2 -> detailed batch information)
    @param ini_params:str   -> loads the first set of params from file, if None ini params are random
    @param par_id:int       -> give id of params to load from file, if None last params are loaded (id = -1)
    @param maxiter:int      -> number of times Adam is re-called per batch (0 -> Adam is called ONCE)
    return -> param_history, loss_hist, params (last params)
    """
    if fname != "tmp":
        assert not os.path.exists(fname)
    t0 = time.time()
    train_xyz, train_energies, val_xyz, val_energies = dataloader(fname=datafile, n_samples=n_samples)

    __m = molsys.mol.from_file("methanol.mfpx")
    bondtable = jnp.array(__m.get_conn_as_tab())
    conn_table = __m.conn
    angletable = gen_angle_table(conn_table)
    dihedraltable = gen_dihedral_table(conn_table)
    ric_tables = (bondtable, angletable, dihedraltable)
    train_rics = mbatch_RICs(train_xyz, ric_tables)
    val_rics = mbatch_RICs(val_xyz, ric_tables)

    t1 = time.time()
    print(f"Loading done after {(t1-t0)} sec")
    if ini_params == None:
       params = init_network_params(layer_sizes, random.PRNGKey(seed))
    else:
        if par_id == None:
            par_id = -1
        with open(f"{ini_params}", "rb") as f:
            params = pickle.load(f)[par_id]
    zero_params = init_zero_params(layer_sizes) #ADAM momenta
    m,v = zero_params, zero_params
    b1 = 0.9
    b2 = 0.999

    param_history = [params]
    loss_hist = []
    max_epoch = 0
    t2 = time.time()
    print(f"Network initiation done after {t2-t1} sec")
    for batchsize, n_epoch in batch:
        t = 0
        # batching
        n_batches = n_samples/batchsize
        train_batch_energies = jnp.array_split(train_energies, n_batches)
        for epoch in range(n_epoch):
            time_start = time.time()
            for j, ref_batch_e in enumerate(train_batch_energies):
                rics = train_rics[j*batchsize:batchsize*j+batchsize]
                #t = 0 #move outside?
                t, m, v, params, loss_value = ADAM_SD(params, rics, ref_batch_e, t = t, m = m, v = v,
                                                    b1 = b1, b2 = b2, a = a)
            #Validation
            val_loss = mbatch_loss(params, val_rics, val_energies)
            batch_loss = mbatch_loss(params, rics, ref_batch_e)
            loss_hist.append([float(batch_loss), float(val_loss)])
            if (epoch+max_epoch) == 1680:   
                param_history.append(params) # change to reduce memory use ...

            # shuffle all trainig-data for next epoch
            key = jax.random.PRNGKey(seed)
            train_rics = random.permutation(key = key, x = train_rics, axis = 0, independent = False)
            train_energies = random.permutation(key = key, x = train_energies, axis = 0, independent = False)
            train_batch_energies = jnp.array_split(train_energies, n_batches)
            time_end = time.time()
            print(f"Train: {batch_loss:15.12f} \t Val: {val_loss:15.12f} \t Epoch: {epoch+max_epoch:6.0f} in {round(time_end-time_start, 3):6.3} sec")
        max_epoch += n_epoch
    param_history.append(params)
    t3 = time.time()
    # save params and loss_hist and generate out file ----------------------- remove from training func?
    now = datetime.now()
    now_string = now.strftime("%d/%m/%Y %H:%M:%S")
    info = f"""
    Calculation finished on: {now_string} with a total run time of {(t3-t0)/60} minutes
    Params loaded from {ini_params} with id {par_id}
    Seed: {seed}
    Hyperparameter:
    \t number of trainig data {n_samples}
    \t number of validation data {len(val_rics)}
    \t number of hidden layers {len(layer_sizes)-2} with the layer structure {layer_sizes} input + hidden + [...] + out
    \t batch size and number of epochs run {batch}
    \t number of epochs {max_epoch}
    \t Adam used params: b1 = {b1}, b2 = {b2}, a = {a}, maxiter = {maxiter}
    
    Last loss values: Training: {batch_loss} \t Validation: {val_loss}
    Min loss value Training: {jnp.min(jnp.array(loss_hist).T[0])} in epoch {jnp.argmin(jnp.array(loss_hist).T[0])}
    Min loss value Validation: {jnp.min(jnp.array(loss_hist).T[1])} in epoch {jnp.argmin(jnp.array(loss_hist).T[1])}
    History of params and loss values can be found in param_hist_{fname}.pkl and loss_hist_{fname}.pkl respectivly

    Note: info about the used activation function is not yet saved, default is jax.nn.GeLu
    """
    with open(fname+".out", "w") as f:
        f.write(info)    
    with open(f"param_hist_{fname}.pkl", "wb") as f:
        pickle.dump(param_history, f)
    with open(f"loss_hist_{fname}.pkl", "wb") as f:
        pickle.dump(loss_hist, f)  


    if create_plot == True:
        plt.figure(f"{fname}")                               
        plt.plot(np.arange(len(loss_hist)), np.array(loss_hist).T[0], label=f"training")
        plt.plot(np.arange(len(loss_hist)), np.array(loss_hist).T[1], label=f"validation")
        #plt.plot(np.arange(len(loss_hist)), np.array([RMS]*len(loss_hist)), label="RMS")
        plt.legend()
        plt.title(f"a: {a}; maxiter: {maxiter}; min loss: {round(np.min(np.array(loss_hist).T[0]), 5)}\n {layer_sizes}")
        plt.xlabel("Epoch")
        plt.ylabel("RMS Loss")
        if save_plot == True:
            plt.savefig(f"{fname}.png")
        else:
            plt.show()   

    return

if __name__ == "__main__":
    t0 = time.time()
    #jax.default_device = jax.devices("gpu")[0]
    training(fname = "tmp", datafile = "trainingsdata/3000K_methanol.hdf5", 
             layer_sizes = [15, 1200, 800, 1200, 1], a = 1/10000, n_samples = 15000, 
             batch = [(10, 200), (200, 10000), (15000, 10000)], 
             create_plot = True, save_plot = True, verbose = None,
             ini_params = None, par_id = None, maxiter = None, seed = 0)
    t1 = time.time()
    print(f"Done after {(t1-t0)/60} min")
