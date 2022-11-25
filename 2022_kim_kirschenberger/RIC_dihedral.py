######################################################################
# function to generate dihedrals of a system for the purpose of NN
# utilising JAX
#
# !!!All inputs must be converted to jax.nump.array!!!
######################################################################

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, grad


if __name__ == "__main__":
    import molsys
    m = molsys.mol.from_file("methanol.mfpx")
    m.detect_conn()
    bondtable = jnp.array(m.get_conn_as_tab())
    xyz = jnp.array(m.get_xyz())
    conn_table = m.conn

def gen_dihedral_table(conn_table):
    """
    takes a mol.conn list an generates an array shape (x, 4)
    plane1|plane2|plane3|dihedral
    @param conn_table: molsys.mol.conn type
    """
    dihedral_table = jnp.empty(shape = (0,4), dtype = np.int16)
    for i, con in enumerate(conn_table):
        for j in range(i+1, len(conn_table)):
            if i in conn_table[j]:
                #check for any new entries in j
                diff = list(set(conn_table[j])-set(con)-set([i]))
                if diff != []:
                    if len(diff) == 1:
                        delta = list(diff)
                    else:
                        delta = diff
                    var_set = con
                    var_set.remove(j)
                    for var in var_set:
                        for angle_atom in delta:
                            tmp = [i, j, var, angle_atom]
                            tmp_array = jnp.array(tmp)[np.newaxis,:]
                            dihedral_table = jnp.append(dihedral_table, tmp_array, axis = 0)
    return dihedral_table

#@jit
def dihedral(xyz, quart):
    """
    takes an entry from diehedraltable and returns a dihedral angle
    @param xyz: whole coords of the current strucutre
    return: angle in rad
    """
    c,b,d,a = quart #bond situation: a-b-c-d -> dihedral between a,d is calculated
    
    vec1 = xyz[a]-xyz[b] #plane 1
    vec2 = xyz[c]-xyz[b] #plane 1,2
    vec3 = xyz[d]-xyz[c] #plane 2

    v1xv2 = jnp.cross(vec1,vec2)
    v2xv3 = jnp.cross(vec2,vec3)

    y = jnp.dot(jnp.cross(v1xv2,v2xv3),vec2/jnp.linalg.norm(vec2))
    x = jnp.dot(v1xv2,v2xv3)
    phi = jnp.arctan2(y,x)
    fac = jnp.sign(phi)
    return jnp.pi - fac * phi

#@jit
def all_dihedral(xyz, dihedraltable):
    """
    vmap wrap of the dihedral function
    @param xyz(jnp.array): the whole xyz coord array
    @param dihedraltable(jnp.array): whole dihedraltable

    return (jnp.array): dihedralangles
    """
    return vmap(fun = dihedral, in_axes = (None, 0))(xyz, dihedraltable)

#@jit
def grad_dihedral(xyz, quart):
    """
    function to calculate the gradient for dihedral in regards to the params
    """
    return grad(dihedral)(xyz, quart)

#@jit
def all_grad_dihedral(xyz, dihedraltable):
    """
    vmap wrapper for the grad_dihedral function
    """
    return vmap(fun= grad_dihedral, in_axes = (None, 0))(xyz, dihedraltable)


if __name__ == "__main__":
    di = gen_dihedral_table(conn_table)
    print(all_dihedral(xyz, di)/(jnp.pi*2)*360)
