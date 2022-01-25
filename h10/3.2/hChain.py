import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
import prepVMC

r = 3.2
atomstring = ""
for i in range(10):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(
    atom=atomstring,
    basis='ccpvdz',
    verbose=4,
    unit='bohr',
    symmetry=0,
    spin=0)
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao

# uhf
dm = [numpy.zeros((norb, norb)), numpy.zeros((norb, norb))]
for i in range(mol.nao//10):
  dm[0][10*i, 10*i] = 1.
  dm[1][10*i+1, 10*i+1] = 1.
umf = prepVMC.doUHF(mol, dm)

# integrals in the canonical orbital basis
mc = mcscf.CASSCF(mf, 10, 10)
mc.kernel()

norbAct = 10
moAct = mc.mo_coeff[:, :norbAct]
h1 = moAct.T.dot(mf.get_hcore()).dot(moAct)
eri = ao2mo.kernel(mol, moAct)
tools.fcidump.from_integrals('FCIDUMP_can', h1, eri, norbAct, mol.nelectron, mf.energy_nuc())

# set up dqmc calculation
rhfCoeffs = numpy.eye(norb)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")
overlap = mf.get_ovlp(mol)
uhfCoeffs = numpy.empty((norb, 2*norb))
uhfCoeffs[::,:mol.nao] = prepVMC.basisChange(umf.mo_coeff[0], mf.mo_coeff, overlap)
uhfCoeffs[::,mol.nao:] = prepVMC.basisChange(umf.mo_coeff[1], mf.mo_coeff, overlap)
prepVMC.writeMat(uhfCoeffs, "uhf.txt")

# cholesky integrals using this funtion in pauxy
h1e, chol, nelec, enuc = prepVMC.generate_integrals(mol, mf.get_hcore(), mc.mo_coeff, chol_cut=1e-5, verbose=True)

nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0
chol = chol.reshape((chol.shape[0], -1))
prepVMC.write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, filename='FCIDUMP_chol')

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = 0
mycc.verbose = 5
mycc.kernel()
#prepVMC.write_ccsd(mycc.t1, mycc.t2)

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
