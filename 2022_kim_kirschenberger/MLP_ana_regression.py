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
from datetime import datetime
import pickle

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
        #curr_layer = jnp.tanh(jnp.dot(w, curr_layer)+b)
        curr_layer = jax.nn.gelu(jnp.dot(w, curr_layer)+b)
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


# generate data for regression:
def func(inp):
    x,y = inp
    return (1 - x / 2 + x**5 + y**3) * jnp.exp(-x**2 -y**2)

def mbatch_func(inp):
    return vmap(func, in_axes = (0))(inp)

def inp_vectors(n):
    """[[x,y],[x,y], ...]"""
    x = np.linspace(-4,4,n)
    xx = np.array(np.meshgrid(x,x))
    xx = xx.reshape(2,n*n)
    xx = xx.T
    return xx


# Main Programm
def main(fname = "tmp", layer_sizes = None, a = None,
         n_samples = None, batch_size = None, n_epoch = None,
         create_plot = None, save_plot = None, verbose = None,
         ini_params = None, par_id = None, maxiter = 0, seed = 0):
    """
    Combining all building blocks to a NN with a training loop
    @param fname:str        -> base for all files generated by the program
    @param layer_sizes:list -> layer geometry, list of all layer sizes including in and out
    @param a:float          -> (initial) step size for ADAM optimizer
    @param n_samples:int    -> number of traing points (val NOT effected!)
    @param batch_size:int   -> size of the individual batches
    @param n_epoch:int      -> max number of epochs
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

    start_time = time.time()
    inp_total = inp_vectors(100)
    ref_total = mbatch_func(inp_total)

    inp_train = inp_total[:n_samples]
    ref_train = ref_total[:n_samples]

    inp_val = np.random.uniform(low = -4, high = 4, size = (n_samples,2))
    inp_val = np.unique(inp_val, axis = 0)
    inp_val = np.array([i for i in inp_val if i not in inp_train])
    ref_val = mbatch_func(inp_val)
    RMS = np.sqrt(np.mean(np.square(ref_train)))
    if verbose > 0:
        print("RMS of traindata", np.sqrt(np.mean(np.square(ref_train))))

    # batching
    b_size = batch_size
    n_batches = n_samples/b_size
    #batches_xyz = jnp.array_split(sample_xyz, n_batches)
    batches_ref_train = jnp.array_split(ref_train, n_batches)

    # Hyperparameters
    #layer_sizes = [2, 55, 25, 10, 1] #--> RIESEN UNTERSCHIED!
    if ini_params == None:
        params = init_network_params(layer_sizes, random.PRNGKey(seed))
    else:
        if par_id == None:
            par_id = -1
        with open(f"{ini_params}", "rb") as f:
            params = pickle.load(f)[par_id]
    zero_params = init_zero_params(layer_sizes) # Adam

    # NN-loop
    load_done = time.time()
    if verbose > 0:
        print(f"Loading completed in {(load_done-start_time)/60} min")
    param_history = []
    param_history.append(params)
    loss_hist = []
    n_epoch = n_epoch

    b1 = 0.9
    b2 = 0.999
    m,v = zero_params, zero_params # reset at all?
    a_flag = True # False
    for epoche in range(n_epoch):
        for j, batch_ref in enumerate(batches_ref_train):
            rics = inp_train[j*b_size:b_size*j+b_size]
            loss_value = 100 # start value > convergenace criteria
            t = 0
            iter = 0
            time_start = time.time()
            while True:
                t, m, v, params, loss_value = ADAM_SD(params, rics, batch_ref, t = t, m = m, v = v,
                                                    b1 = b1, b2 = b2, a = a)
                if iter >= maxiter:
                    break
                iter += 1
            time_end = time.time()
            if verbose > 1:
                print(f"Batch {j} of Epoch {epoche} done in {t} Timesteps and in {round(time_end-time_start, 3)} sec")
            
        # Validation batch
        val_loss = mbatch_loss(params, inp_val, ref_val)
        batch_loss = mbatch_loss(params, inp_train, ref_train)
        if verbose > 0:
            print(f"Train: {batch_loss} \t Val: {val_loss} \t Epoch: {epoche}")
        loss_hist.append([float(batch_loss), float(val_loss)])    
        param_history.append(params)

        # shuffle all trainig-data for next epoch
        key = jax.random.PRNGKey(42)
        inp_train = random.permutation(key = key, x = inp_train, axis = 0, independent = False)
        ref_train = random.permutation(key = key, x = ref_train, axis = 0, independent = False)
        batches_ref_train = jnp.array_split(ref_train, n_batches)

        #if epoche in [1000, 10000] and a_flag == False:    --> not properly implemented yet
        #    a = a/10
        #    a_flag = False
        #    if verbose > 0:
        #        print("Adam stepsize decreased, doing more optimizer cycles now")
    
    end_time = time.time()
    if verbose > 0: 
        print(f"Done after {(end_time-start_time)/60} min")

    # save params and loss_hist and generate out file
    now = datetime.now()
    now_string = now.strftime("%d/%m/%Y %H:%M:%S")
    info = f"""
    Calculation finished on: {now_string} with a total run time of {(end_time-start_time)/60} minutes
    Params loaded from {ini_params} with id {par_id}
    Seed: {seed}
    Hyperparameter:
    \t number of trainig data {n_samples}
    \t number of validation data {n_samples}
    \t number of hidden layers {len(layer_sizes)-2} with the layer structure {layer_sizes} input + hidden + [...] + out
    \t batch size {batch_size}
    \t number of epochs {n_epoch}
    \t Adam used params: b1 = {b1}, b2 = {b2}, a = {a}, maxiter = {maxiter}
    
    Last loss values: Training: {batch_loss} \t Validation: {val_loss}
    Min loss value Training: {np.min(np.array(loss_hist).T[0])} in epoch {np.argmin(np.array(loss_hist).T[0])}
    Min loss value Validation: {np.min(np.array(loss_hist).T[1])} in epoch {np.argmin(np.array(loss_hist).T[1])}
    History of params and loss values can be found in param_hist_{fname}.pkl and loss_hist_{fname}.pkl respectivly
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
    return param_history, loss_hist, params


if __name__ == "__main__": 
    # test setup for stepsize
    s = time.time()
    rc('figure', figsize=(8.27,11.69)) # show only last 1000 epochs!
    for layer in [[2,100,100,1]]: #[[2,20,40,1], [2,40,40,20,1], [2,60,40,60,1]]:
        for batchsize in [100]:
            a = 1/10000
            fig, ax = plt.subplots(5,2)
            fig.tight_layout()
            fig.suptitle(f"{layer}_{a}",y=1)
            for i in range(10):
                param_history, loss_hist, params = main(fname = f"batchsize_2_60_40_60_1_{batchsize}_{i}", layer_sizes = layer, a = a,
                                                        n_samples = 10000, batch_size = batchsize, n_epoch = int(10000),
                                                        create_plot = False, save_plot = False, verbose = 1,
                                                        ini_params = None, par_id = None, maxiter = 0, seed = i)
                ax[i//2,i%2].plot(np.arange(len(loss_hist))[-1000:], np.array(loss_hist).T[0,-1000:], label=f"training")
                ax[i//2,i%2].plot(np.arange(len(loss_hist))[-1000:], np.array(loss_hist).T[1,-1000:], label=f"validation")
                if i == 8:
                    ax[i//2,i%2].set_xlabel("epoch")
                    ax[i//2,i%2].set_ylabel("RMS loss")
                if i == 1:
                    ax[i//2,i%2].legend()
            plt.savefig(f"batchsize_2_60_40_60_1_{batchsize}.png", dpi = 500, bbox_inches = "tight")
            #plt.show()
    print(f"total done after {time.time()-s}")

                             
