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
#config.update("jax_enable_x64", True)
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

#@jit
def get_all_RIC_grads(xyz, ric_tables):
    # broadcast to fit iteration cycles with the same RICS, before vmap?
    gb = all_grad_bond(xyz, ric_tables[0])
    ga = all_grad_angle(xyz, ric_tables[1])
    gd = all_grad_dihedral(xyz, ric_tables[2])
    return jnp.concatenate((gb,ga,gd))

@jit
def grad_mbatch_RICs(xyz, ric_tables):
    """vmap wrapper for RIC grad over a mini batch"""
    return vmap(get_all_RIC_grads, in_axes = (0, None))(xyz, ric_tables)

# Core structure and functions
@jit
def MLP(params, inp): # note: still uses a for loop ...
    """doc string"""
    curr_layer = inp
    for w, b in params[:-1]: # only till the last one, because then NO activation function is used!
        curr_layer = jax.nn.gelu(jnp.dot(w, curr_layer)+b)
    w, b = params[-1]
    out_layer = jnp.dot(w, curr_layer) + b
    return out_layer[0] # out is size 1, Energy
@jit
def grad_dMLP_dRIC(params, inp):
    """this is a transform to get the gradient of the MLP (Energy) wrt to the rics"""
    return grad(MLP, 1)(params, inp)
@jit
def grad_mbatch_dMLP_dRIC(params, inp):
    """vmap wrapper for grad_dMLP_dRIC"""
    return vmap(grad_dMLP_dRIC, in_axes = (None, 0))(params, inp)
@jit
def ini_MLP(ric, params, dric_dxyz):
    """
    initializes the MLP with the rics as input vector and the random parameters, 
    feedforward of the layers to compute energy BUT ALSO take the derivative wrt the input
    return (float32 !jax!): energy, dE/dxyz
    """
    dE_dRIC = grad_dMLP_dRIC(params, ric)
    dE_dxyz =  (dE_dRIC[:, jnp.newaxis, jnp.newaxis]*dric_dxyz).sum(axis=0) # jnp.dot(dE_dRIC, dric_dxyz)
    return MLP(params, ric), -1.0*dE_dxyz # FORCES = -GRAD!!!! 
#@jit
def mbatch_ini_MLP(ric, params, dric_dxyz):
    """vmap wrapper for ini_MLP for mini batches"""
    return vmap(ini_MLP, in_axes = (0, None, 0))(ric, params, dric_dxyz)

# Loss function
#@jit
def loss(ric, params, dric_dxyz, ref_forces):
    """rms of the deltas --> for a single data point """
    energy, forces = ini_MLP(ric, params, dric_dxyz)
    delta = forces-ref_forces
    # rms = jnp.sqrt(jnp.mean(delta**2))
    #delta_sum_vec = jnp.sum(delta, axis = 0) # froms the sum over all x y z and return the vector x_tot, y_tot, z_tot
    #delta_mag = jnp.sqrt(jnp.sum(jnp.square(delta), axis = 0)) # calculates the magnitude of the total force vector
    rms = jnp.sqrt(jnp.mean(jnp.square(delta)))
    return rms

def get_delta(ric, params, dric_dxyz, ref_forces, ref_energies):
    energy, forces = ini_MLP(ric, params, dric_dxyz)
    fdelta = forces-ref_forces
    edelta = energy-ref_energies
    return fdelta, edelta

@jit
def mbatch_loss(ric, params, dric_dxyz, ref_forces, ref_energies):
    """
    vmap wrapper for the loss function over a MINI BATCH. Takes the rms of all rms.
    @param ric: all rics for all the structures of the mini batch
    @param params: starting params of the mini batch (same for all -> vmap none)
    @param dirc_dxyz: derivitive for each structure
    @param ref_forces: ref forces for all of the structures
    """
    fdelta, edelta = vmap(get_delta, in_axes = (0, None, 0, 0, 0))(ric, params, dric_dxyz, ref_forces, ref_energies)
    frms = jnp.sqrt(jnp.mean(jnp.square(fdelta)))
    erms = jnp.sqrt(jnp.mean(jnp.square(edelta)))
    total_rms = frms+erms
    return total_rms

def mbatch_force_loss(ric, params, dric_dxyz, ref_forces):
    """
    vmap wrapper for the loss function over a MINI BATCH. Takes the rms of all rms.
    @param ric: all rics for all the structures of the mini batch
    @param params: starting params of the mini batch (same for all -> vmap none)
    @param dirc_dxyz: derivitive for each structure
    @param ref_forces: ref forces for all of the structures
    """
    e,f = mbatch_ini_MLP(ric, params, dric_dxyz)
    fdelta = f-ref_forces
    frms = jnp.sqrt(jnp.mean(jnp.square(fdelta)))
    total_rms = frms
    return total_rms

@jit
def grad_mbatch_loss(ric, params, dric_dxyz, ref_forces, ref_energies):
    return grad(mbatch_loss, 1)(ric, params, dric_dxyz, ref_forces, ref_energies)

# Optimizer
@jit
def SD_mbatch_loss(ric, params, dric_dxyz, ref_forces, stepfaktor):
    """not functional"""
    step_size = jnp.array(0.1*stepfaktor, dtype = jnp.float64)
    #loss, grads = value_and_grad(mbatch_loss)(ric, params, dric_dxyz, ref_forces)
    grads = grad_mbatch_loss(ric, params, dric_dxyz, ref_forces)
    loss = mbatch_loss(ric, params, dric_dxyz, ref_forces)
    return [(w + step_size * dw, b + step_size * db)
          for (w, b), (dw, db) in zip(params, grads)], loss

@jit
def ADAM_SD(params, rics, dric_dxyz, ref_forces, ref_energies, t = 0, m = None, v = None, b1 = 0.9, b2 = 0.999, a = 0.001):
    """ 
    Adam Optimizer (doi: https://doi.org/10.48550/arXiv.1412.6980)
    """
    #print(t, m, v)
    # params adapted for different systems
    e = 10**(-8)
    
    grad = grad_mbatch_loss(rics, params, dric_dxyz, ref_forces, ref_energies) # grad wrt params of loss

    t += 1 # m and v musst be the same shape as params ... -> zip + init the same as params!
    m = [(b1 * m_dw + (1-b1) * dw, b1 * m_db + (1-b1) * db) 
         for (dw, db), (m_dw, m_db) in zip(grad, m)] # m = b1 * m + (1-b1) *  grad
    #print("m: ",m)
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
    #print("params: ", params)
    loss = mbatch_loss(rics, params, dric_dxyz, ref_forces, ref_energies)
    return t, m, v, params, loss

@jit
def AdaMax(params, rics, dric_dxyz, ref_forces, t = 0, m = None, u = None, a = 0.002, b1 = 0.9, b2= 0.999):
    """
    AdaMAx optimizer (doi: https://doi.org/10.48550/arXiv.1412.6980)
    approx. same opti step time as adam
    currently: does not change params and is at some point just jnp.nan
    """
    t += 1
    grad = grad_mbatch_loss(rics, params, dric_dxyz, ref_forces) # grad wrt params of loss

    m = [(b1 * m_dw + (1-b1) * dw, b1 * m_db + (1-b1) * db) 
         for (dw, db), (m_dw, m_db) in zip(grad, m)] # m = b1 * m + (1-b1) *  grad

    #u = [(max(max(b2*_uw.flatten()), max(abs(dw.flatten()))), max(max(b2*_ub.flatten()), max(abs(db.flatten()))))
    #     for (dw,db), (_uw, _ub) in zip(grad, u)] # max(b2*u, |grad|)
    u = [(jnp.maximum(b2*_uw, jnp.abs(dw)), jnp.maximum(b2*_ub, jnp.abs(db)))
         for (dw,db), (_uw, _ub) in zip(grad, u)] # max(b2*u, |grad|)

    params = [(w - (a/(1-jnp.power(b1, t))*_mw/_uw), b - (a/(1-jnp.power(b1, t))*_mb/_ub))
              for (w,b), (_mw, _mb), (_uw, _ub) in zip(grad, m, u)]
    loss = mbatch_loss(rics, params, dric_dxyz, ref_forces)
    #print(t, loss)
    return t, m, u, params, loss


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
    loads forces and strucutures from ref file 
    """
    r = refdata(fname = fname)
    all_xyz, all_id = r.get_xyz_from_sample(sample = "MD")
    _max = len(all_xyz)
    r.change_lot("TPSS")
    sample_DFT_forces = r.get_forces_from_id([x for x in range(n_samples)])
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
    sample_FF_forces = r.get_forces_from_id([x for x in range(n_samples)])
    train_forces = sample_DFT_forces - sample_FF_forces 
    train_xyz = all_xyz[:n_samples]

    # validation set
    r.change_lot("TPSS")
    val_DFT_forces = r.get_forces_from_id([x for x in range(n_samples,_max)]) # methods only produces "forward" indexing 
    val_DFT_energies = r.get_energies_from_id([x for x in range(n_samples,_max)]) # methods only produces "forward" indexing 
    val_DFT_energies -= dft0_E
    r.change_lot("MOFFF")
    val_FF_forces = r.get_forces_from_id([x for x in range(n_samples,_max)])
    val_forces = val_DFT_forces - val_FF_forces
    val_FF_energies = r.get_energies_from_id([x for x in range(n_samples,_max)])
    val_FF_energies -= ff0_E
    #val_energies = val_FF_energies # direct fit
    val_energies = val_DFT_energies - val_FF_energies # Delta learning 
    val_xyz = all_xyz[n_samples:_max]
    r.close()
    return train_xyz, train_forces, train_energies, val_xyz, val_forces, val_energies



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
    if fname != "ftmp":
        assert not os.path.exists(fname)
    t0 = time.time()
    train_xyz, train_forces, train_energies, val_xyz, val_forces, val_energies = dataloader(fname=datafile, n_samples=n_samples)

    __m = molsys.mol.from_file("methanol.mfpx")
    bondtable = jnp.array(__m.get_conn_as_tab())
    conn_table = __m.conn
    angletable = gen_angle_table(conn_table)
    dihedraltable = gen_dihedral_table(conn_table)
    ric_tables = (bondtable, angletable, dihedraltable)
    train_rics = mbatch_RICs(train_xyz, ric_tables)
    train_grad_rics = grad_mbatch_RICs(train_xyz, ric_tables)
    val_rics = mbatch_RICs(val_xyz, ric_tables)
    val_grad_rics = grad_mbatch_RICs(val_xyz, ric_tables)

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
    for batchsize, n_epoch in batch:
            t = 0
            # batching
            n_batches = n_samples/batchsize
            train_batch_forces = jnp.array_split(train_forces, n_batches)
            train_batch_energies = jnp.array_split(train_energies, n_batches)
            for epoch in range(n_epoch):
                time_start = time.time()
                for j, (ref_batch_f, ref_batch_e) in enumerate(zip(train_batch_forces, train_batch_energies)):
                    rics = train_rics[j*batchsize:batchsize*j+batchsize]
                    dric_dxyz = train_grad_rics[j*batchsize:batchsize*j+batchsize]

                    t, m, v, params, loss_value = ADAM_SD(params, rics, dric_dxyz, ref_batch_f, ref_batch_e, t = t, m = m, v = v,
                                                        b1 = b1, b2 = b2, a = a)
                #Validation
                val_loss = mbatch_loss(val_rics, params, val_grad_rics, val_forces, val_energies)
                batch_loss = mbatch_loss(train_rics, params, train_grad_rics, train_forces, train_energies)
                loss_hist.append([float(batch_loss), float(val_loss)])
                time_end = time.time()
                print(f"Train: {batch_loss:15.12f} \t Val: {val_loss:15.12f} \t Epoch: {epoch+max_epoch:6.0f} in {round(time_end-time_start, 3):6.3} sec")
                if (epoch+max_epoch) == 1680:   
                    param_history.append(params) # change to reduce memory use ...

                # shuffle all trainig-data for next epoch
                key = jax.random.PRNGKey(seed)
                train_rics = random.permutation(key = key, x = train_rics, axis = 0, independent = False)
                train_grad_rics = random.permutation(key = key, x = train_grad_rics, axis = 0, independent = False)
                train_forces = random.permutation(key = key, x = train_forces, axis = 0, independent = False)
                train_energies = random.permutation(key = key, x = train_energies, axis = 0, independent = False)
                train_batch_forces = jnp.array_split(train_forces, n_batches)
                train_batch_energies = jnp.array_split(train_energies, n_batches)
                time_end = time.time()
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
    training(fname = "ftmp", datafile = "trainingsdata/3000K_methanol.hdf5", 
             layer_sizes = [15, 2000, 2000, 1], a = 1/10000, n_samples = 15000, 
             batch = [(10, 200), (200, 10000), (15000, 10000)], 
             create_plot = True, save_plot = True, verbose = None,
             ini_params = None, par_id = None, maxiter = None, seed = 0)
    t1 = time.time()
    print(f"Done after {(t1-t0)/60} min")

