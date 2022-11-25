######################################################################
# function to generate bondlengths of a system for the purpose of NN
# utilising JAX
# it is assumed that the bondtable can already be found in the refdata
#
# !!!All inputs must be converted to jax.nump.array!!!
######################################################################

import jax.numpy as jnp
from jax import jit, vmap, grad


if __name__ == "__main__":
    import molsys
    #m = molsys.mol.from_file("methanol.mfpx")
    #m.detect_conn()
    #bondtable = jnp.array(m.get_conn_as_tab())
    #xyz = jnp.array(m.get_xyz())
    __m = molsys.mol.from_file("methanol.mfpx")
    conn_table = jnp.array(__m.get_conn_as_tab())
    bondtable = __m.conn
    print(bondtable)


#@jit
def bond(xyz, i_bond):
    """
    @param xyz(jnp.array): the whole xyz coord array
    @param bond(jnp.array): single row entry from the bondtable
    return: bondlength
    """
    i,j = i_bond
    r = xyz[i]-xyz[j]
    dist = jnp.linalg.norm(r)
    return dist

#@jit
def all_bond(xyz, bondtable):
    """
    vmap wrap of the bondlenght function
    @param xyz(jnp.array): the whole xyz coord array
    @param bondtable(jnp.array): whole bondtable
    return (jnp.array): bonds
    """
    return vmap(fun = bond, in_axes = (None, 0))(xyz, bondtable)

#@jit
def grad_bond(xyz, i_bond):
    """
    function to generate the gradient of the bond function in regard to its parameters
    """
    return grad(bond)(xyz,i_bond)

#@jit
def all_grad_bond(xyz, bondtable):
    """
    vmap warp of the grad_bond function, to get the gradient over all bonds
    """
    return vmap(fun = grad_bond, in_axes = (None, 0))(xyz, bondtable)


if __name__ == "__main__":
    print(bond(xyz, bondtable[0]))
    print(all_bond(xyz, bondtable))
    print(grad_bond)
    print(all_grad_bond(xyz, bondtable))
    
