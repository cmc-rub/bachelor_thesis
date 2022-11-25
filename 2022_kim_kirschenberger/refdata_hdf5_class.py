"""
        refdata

a new class to replace refclass, holding reference info for fitting

- for a given system (N atoms) it contains M structures which can be extended.
- in groups it contains computed data (one group per level of theory => multilevel input is possible)
- data can be energies forces hessians (or other things like charges) refernced ot the structure index
      => not all properties available for all structures
- level of theory stored in metadata attributes
- a strategy info is stored how each structure was sampled (by MD, by random sampling ...)
      => this can be used to split structures and properties in fit/test sets etc.

input and ouptut is as much as possible handled by molsys objects and its addons.

root groups are:
/system (elems, bcond)
/struc (xyz, [cell], sample) 

further groups named by level of theory. three are only conventions but no fixed rules.
eg.
/dft
=> attributes contain metadata .. defined attribute is code .. all the rest defined by the content of code

   subgroups with properties
   /e /ei    # energy and index
   /f /fi    # force and index
   /h /hi    # hessian and index

UNITS:
we do not mess around with different units.
all length are in Angstrom
all energies in kcal/mol
charges in electrons 
etc.
=> convert all things when adding to refdata

RS RUB 2021

"""

from typing import Iterable
import numpy as np
import os
import h5py

import molsys

allowed_codes = ["TM", "ORCA", "MOFFF"]  # list of code names allowed in the refdata .. TM = turbomole

class refdata:

    def __init__(self, fname, mol=None, read_only=False, mfpx_file = None, default_inp_file = None):
        """generate a refdata object/file

        If mol is given then the fname should not exist and is generated
        if mol is not given then the system is opened from the file

        In mol only xyz and elems are needed .. so reading an xyz is fine
        all the other info is not stored in the refdata anyway (we assume QM refdata)
    
        Args:
            fname ([string], optional): Filename of hdf5 file.
            mol ([molsys object], optional): When given generate a new file. Defaults to None.
        """

        self.charge = "None"
        self.mulitplicity = "None"
        self.default_inp_file = default_inp_file

        if isinstance(mol, molsys.mol) == True:
            # Make a new file from mol object
            assert not os.path.exists(fname), "hdf5 file should not exist"
            self.f = h5py.File(fname, "w")  
            self.system = self.f.create_group("system")
            self.struc  = self.f.create_group("struc")
            # set up system
            self.elems = self.system.create_dataset("elems", data=mol.get_elems())
            self.bcond = mol.get_bcond()
            self.system.attrs["bcond"] = self.bcond

            self.system.attrs["charge"] = self.charge
            self.system.attrs["multiplicity"] = self.mulitplicity
            # set up structure
            self.na = mol.get_natoms()
            self.xyz = self.struc.create_dataset("xyz", (0, self.na, 3), chunks=(1, self.na, 3), maxshape=(None, self.na, 3), dtype="float64")
            # self.xyz[0,::] = mol.get_xyz() --> trying to load it with the first LOT data
            if self.bcond > 0:
                self.cell = self.struc.create_dataset("cell", (1, 3, 3), chunks=(1, 3, 3), maxshape=(None, 3, 3), dtype="float64")
                self.cell[0, ::] = mol.get_cell()
            self.sample = self.struc.create_dataset("sample", (0,), maxshape=(None,), dtype="int32", track_order=True) # track_order=True makes attrs to remeber the order
            #self.sample[0] = 0
            #self.sample.attrs["initial"] = 0
            self.read_only = False
            # defaults
            self.curr_lot = None


        elif mfpx_file != None:
            assert not os.path.exists(fname), "hdf5 file should not exist"

            m = molsys.mol.from_file(mfpx_file)
            # create hdf5 file WITHOUT an existing molsys object instead gen from mfpx
            self.f = h5py.File(fname, "w")
            self.system = self.f.create_group("system")
            self.struc  = self.f.create_group("struc")

            # --> system
            self.elems = self.system.create_dataset("elems", data=m.get_elems())
            self.bcond = m.get_bcond()
            self.system.attrs["bcond"] = self.bcond

            self.system.attrs["charge"] = self.charge
            self.system.attrs["multiplicity"] = self.mulitplicity
            
            # --> structure
            self.na = m.get_natoms()
            self.xyz = self.struc.create_dataset("xyz", (0, self.na, 3), chunks=(1, self.na, 3), maxshape=(None, self.na, 3), dtype="float64")
            #self.xyz[0,::] = m.get_xyz()
            if self.bcond > 0:
                self.cell = self.struc.create_dataset("cell", (1, 3, 3), chunks=(1, 3, 3), maxshape=(None, 3, 3), dtype="float64")
                self.cell[0, ::] = m.get_cell()
            self.sample = self.struc.create_dataset("sample", (0,), maxshape=(None,), dtype="int32", track_order=True) # track_order=True makes attrs to remeber the order
            #self.sample[0] = 0
            #self.sample.attrs["initial"] = 0
            self.read_only = False

            # --> defaults
            self.curr_lot = None


        elif mol == None and mfpx_file == None and fname != None:
            # Read from exisiting file .. take the init structure 0 as the current
            assert os.path.exists(fname)
            if read_only:
                self.read_only = True
                self.f = h5py.File(fname, "r")
            else:
                self.f = h5py.File(fname, "r+")
                self.read_only = False
            self.system = self.f["system"]
            self.struc  = self.f["struc"]
            self.elems  = self.system["elems"]
            self.bcond  = self.system.attrs["bcond"]

            self.charge = self.system.attrs["charge"]
            self.mulitplicity = self.system.attrs["multiplicity"]
            
            self.xyz    = self.struc["xyz"]
            self.na     = self.xyz.shape[1] # --> does this work without a given xyz entry in the beginning?
            if self.bcond > 0:
                self.cell = self.struc["cell"]
            self.sample = self.struc["sample"]
            # set the curr_lot (level of theory) to the first we find .. if there is one all is ok anyway
            self.change_lot(self.get_lots()[0])
        else:
            raise Exception("No System data could be generated or read from file. Give a mol object, mfpx file or and an existing hdf5 of this classes structure")
        return

    #### handle levels of theory

    def set_new_lot(self, name, code, helper):
        """generate a new lot group

        Args:
            name (string): name of the group
            code (string): name of the code (like ORCA or TM), must be one of the known
            helper (class): class reading from the provided output files into a formate excepted by refdata (like refdata_TMhelper.py ...)
        """
        assert self.read_only == False
        # make sure the lot does not exist, yet
        assert name not in self.get_lots()
        assert code in allowed_codes
        if code == "TM":
            meta = helper.get_metadata()
        elif code == "ORCA":
            meta = helper.get_metadata(self.default_inp_file)
        elif code == "MOFFF":
            meta = {"there is nothing to see here ... yet": "0"}
        # now we can make the lot and add the attributes
        self.curr_lot = self.f.create_group(name)
        for k in meta:
            self.curr_lot.attrs[k] = meta[k]
        # now set up some structures
        egroup = self.curr_lot.create_group("energy")
        egroup.create_dataset("data", (0,), maxshape=(None,), dtype="float64")
        egroup.create_dataset("index", (0,), maxshape=(None,), dtype="int32")
        fgroup = self.curr_lot.create_group("force")
        fgroup.create_dataset("data", (0, self.na, 3), maxshape=(None, self.na, 3), chunks=(1, self.na, 3), dtype="float64")
        fgroup.create_dataset("index", (0,), maxshape=(None,), dtype="int32")
        hgroup = self.curr_lot.create_group("hessian")
        nh = 3*self.na
        hgroup.create_dataset("data", (0, nh, nh), maxshape=(None, nh, nh), chunks=(1, nh, nh), dtype="float64")
        hgroup.create_dataset("index", (0,), maxshape=(None,), dtype="int32")
        return

    def get_lots(self):
        """get all existing lots

        Returns:
            list of strings: names of existing levels of theory
        """
        lots = list(self.f.keys())
        lots.remove('system')
        lots.remove('struc')
        #print(lots)
        return lots

    def change_lot(self, new_lot):
        assert new_lot in self.get_lots()
        self.curr_lot = self.f[new_lot]
        return


    #### add new data to the file 

    def add_energy_force_xyz(self, x, e, f, charge:int = None, multiplicity:int = None ,sample=None, add=False, slice=None):
        """add points on the PES to the reference (for the current level of theory!!)

        this can either mean adding energies/forces for exisiting structures sampled somehow or
        adding new structures.
        If add==True you need to provide sample.
        If add==False slice is needed and must match the number of new points.

        TODO: a lot of testing: if you set for exisiting structures make sure these exist
              test if the geometry is equal to the registered one
              etc.
              currently this is minimalisitc assuming you do all in the right way!!!

        Args:
            x (numpy.darray): positional data
            e (numpy.darray): energies (in kcal/mol)
            f (numpy.darray): forces (in kcal/mol/A)
            slice (tuple of ints, optional): [description]. Defaults to None.
            add (bool, optional): if True add new points, if False set only energies and forces. Defaults to False.
        """
        assert self.read_only == False
        if self.charge != "None":
            assert self.charge == charge, f"The charge is not {self.charge} and therefore not the same as for the the other structures!"
        elif charge != None:
            self.charge = charge
            self.system.attrs["charge"] = self.charge

        if self.mulitplicity != "None":
            assert self.mulitplicity == multiplicity, f"The multiplicity is not {self.mulitplicity} and therefore not the same as for the other structures!"
        elif multiplicity != None:
            self.mulitplicity = multiplicity
            self.system.attrs["mulitlicity"] = self.mulitplicity
        
        egroup = self.curr_lot["energy"]
        ed = egroup["data"]
        ei = egroup["index"]
        fgroup = self.curr_lot["force"]
        fd = fgroup["data"]
        fi = fgroup["index"]
        xyz    = self.struc["xyz"]
        if add:
            # add new data
            # check if sample strategy alredy exists, if not add it
            assert sample != None
            if sample not in self.sample.attrs.keys():
                nsample = 0
                for s in self.sample.attrs:
                    if self.sample.attrs[s] > nsample:
                        nsample = self.sample.attrs[s]
                nsample += 1
                self.sample.attrs[sample] = nsample
            else:
                nsample = self.sample.attrs[sample]
            # now increase fields and add
            # TODO assert that sizes match .. currentyl we just get eerors without info
            #check if iterable data is provided aka multiple sets of energy
            if isinstance(e, Iterable) == True: # old if type(e) == np.float64
                nnew = len(e)
            else:
                nnew = 1
            nx = len(xyz)
            #print(f"Lenght of xyz before any data is added: {nx}")
            #if nx == 1: #no need to resize if 
            #    xyz.resize((nx, self.na, 3))
            #    self.sample.resize((nx,))
            #else:
            #    xyz.resize((nx+nnew, self.na, 3))
            #    self.sample.resize((nx+nnew,))
            
            xyz.resize((nx+nnew, self.na, 3))
            self.sample.resize((nx+nnew,))

            self.sample[nx:] = nsample
            ne = len(ed)
            ed.resize((ne+nnew,))
            ei.resize((ne+nnew,))
            nf = len(fd)
            fd.resize((nf+nnew, self.na, 3))
            fi.resize((nf+nnew,))
            xyz[nx:] = x
            ed[ne:]  = e
            ei[ne:]  = np.arange(nx, nx+nnew ,dtype="int32")
            fd[nf:]  = f
            fi[nf:]  = np.arange(nx, nx+nnew ,dtype="int32")
            return (nx+nnew, nnew)
        else:
            # structures exist but we still need to extend e and f
            nnew = slice[1]-slice[0]
            ne = len(ed)
            ed.resize((ne+nnew, ))
            ei.resize((ne+nnew, ))
            nf = len(fd)
            fd.resize((nf+nnew, self.na, 3))
            fi.resize((nf+nnew,))
            if nnew > 1:
                for i in range(slice[0], slice[1]):
                    j = i-slice[0]
                    d = xyz[i]-x[j]
                    msd = (d*d).sum()/(self.na*3)
                    assert msd < 1e-10, "Geometry does not match registered geom %d (msd = %10.5f)" % (i, msd)
                    # ok add values
                    ed[ne+j-1] = e[j]
                    ei[ne+j-1] = i
                    fd[nf+j-1] = f[j]
                    fi[ne+j-1] = i
            else:
                d = xyz[slice[0]]-x
                msd = (d*d).sum()/(self.na*3)
                assert msd < 1e-10, "Geometry does not match registered geom %d (msd = %10.5f)" % (slice[0], msd)
                ed[ne] = e
                ei[ne] = slice[0]
                fd[nf] = f
                fi[nf] = slice[0]
        return 

    def add_hessian(self, h, i):
        """add hessian info to file

        TODO: provide xyz coords (optional?) to test if the structure is the right one
        Hessian is costly so we do not add multiple but always one at a time.
        Structures must exist.

        Args:
            h (numpy darray 2d): Hessian array in kcal/mol/A^2
            i (int): structure to which this Hessian belongs
        """
        assert self.read_only == False
        assert h.shape == (3*self.na, 3*self.na)
        assert i < self.xyz.shape[0]
        hgroup = self.curr_lot["hessian"]
        hd = hgroup["data"]
        hi = hgroup["index"]
        nh = hd.shape[0]
        hd.resize((nh+1, 3*self.na, 3*self.na))
        hi.resize((nh+1,))
        hd[nh] = h
        hi[nh] = i
        return

    ### other things ###

    def get_lowest_energy(self):
        """returns index of lowest energy of current lot
        """
        egroup = self.curr_lot["energy"]
        ed = egroup["data"]
        ei = egroup["index"]
        jmin = np.argmin(ed)
        return ei[jmin]        

    ### get data from ref

    def get_xyz(self, idx):
        if idx == "lowest":
            idx = self.get_lowest_energy()
        # print ("getting xyz %d" % idx)
        return np.array(self.xyz[idx])

    def get_xyz_from_sample(self, sample):
        """return a series for xy coords for a given sampling strategy
        (and the current lot of course)

        Args:
            sample (string): sampling type
        """
        nsample = self.sample.attrs[sample]
        idx = np.where(self.sample == nsample)
        return self.xyz[idx], idx

    def get_energies_from_id(self, idx):
        ei = self.curr_lot["energy/index"]
        ed = self.curr_lot["energy/data"]
        ids = np.searchsorted(ei,idx)
        energies = ed[ids]
        return energies

    def get_forces_from_id(self, idx):
        fi = self.curr_lot["force/index"]
        fd = self.curr_lot["force/data"]
        ids = np.searchsorted(fi,idx)
        forces = fd[ids]
        return forces
        
    def get_hessian(self, idx):
        if idx == "lowest":
            idx = self.get_lowest_energy()
        # cdprint ("getting hessian %d" % idx)
        hgroup = self.curr_lot["hessian"]
        hd = hgroup["data"]
        hi = hgroup["index"]
        lhi = list(hi)
        j = lhi.index(idx)
        return np.array(hd[j])        

    def close(self):
        self.f.close()


if __name__ == "__main__":
    test  = refdata(fname = "test_refdata.hdf5", mfpx_file = "cupwph4.mfpx")



