######################################################################
# function to generate bondangles of a system for the purpose of NN
# utilising JAX
#
# !!!All inputs must be converted to jax.nump.array!!!
######################################################################

import jax.numpy as jnp
from jax import jit, vmap, grad
from itertools import permutations


if __name__ == "__main__":
    import molsys
    m = molsys.mol.from_file("methanol.mfpx")
    m.detect_conn()
    bondtable = jnp.array(m.get_conn_as_tab())
    xyz = jnp.array(m.get_xyz())
    conn_table = m.conn

def gen_angle_table(conn_table):
    """
    take a mol.conn list and generates an array containing 3 atoms per row:
    c|A|B with c as connecting atom and the vectors cA cB forming the angle
    @param conn_table: molsys.mol.conn type
    """
    angle_array = jnp.empty(shape = (0,3), dtype=jnp.int16)
    for i, con in enumerate(conn_table):
        if len(con) > 1:
            perm = list(permutations(con, 2))
            perm = [sorted(list(x)) for x in perm]
            for sublist in perm:
                sublist.insert(0,i)
            perm = jnp.unique(jnp.array(perm), axis = 0)
            angle_array = jnp.append(angle_array, perm, axis = 0)
    #print("angletable", angle_array)
    return angle_array

#@jit
def angle(xyz, trio):
    """
    Calculates the angle for a given entry of the angle table
    @param xyz: whole coords of the current strucutre
    @param angle: angle table generated by gen_angle_table
    return: angle in rad
    """
    c, a, b = trio
    vec1 = xyz[a]-xyz[c]
    vec2 = xyz[b]-xyz[c]
    dot_12=jnp.dot(vec1, vec2)
    norm_vec1 = jnp.linalg.norm(vec1)
    norm_vec2 = jnp.linalg.norm(vec2)

    angle_rad = jnp.arccos(dot_12/(norm_vec1*norm_vec2))
    return angle_rad

#@jit
def all_angle(xyz, angletable):
    """
    vmap wrap of the bondangle function
    @param xyz(jnp.array): the whole xyz coord array
    @param angletable(jnp.array): whole angletable

    return (jnp.array): bondangles
    """
    return vmap(fun = angle, in_axes = (None, 0))(xyz, angletable)

#@jit
def grad_angle(xyz, trio):
    """
    calculates the gradient of the angle function in regards to its params
    """
    return grad(angle)(xyz, trio)


def all_grad_angle(xyz, angletable):
    """
    vmap wrapper fpr grad_angle, to get the gradient of all angles
    """
    return vmap(fun = grad_angle, in_axes = (None, 0))(xyz, angletable)

    

if __name__ == "__main__":
    angletable = gen_angle_table(conn_table)
    print(angletable)
    
