import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
from pauxy.utils.from_pyscf import generate_integrals
import prepVMC

r = 1.1282
atomstring = f'''
C 0. 0. 0.
O {r} 0. 0.
'''
mol = gto.M(
     atom=atomstring,
     basis='augccpvqz',
     verbose=4,
     symmetry=0,
     spin=0)
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao
nelec = mol.nelec
print(f'norb: {norb}')
print(f'nelec: {nelec}')

mc = mcscf.CASSCF(mf, 8, 10)
mc.kernel()

# including 1s's
norbAct = 50
moAct = mc.mo_coeff[:, :norbAct]

# integrals in the canonical orbital basis
h1 = moAct.T.dot(mf.get_hcore()).dot(moAct)
eri = ao2mo.kernel(mol, moAct)
tools.fcidump.from_integrals('FCIDUMP_can', h1, eri, norbAct, mol.nelectron, mf.energy_nuc())

# set up dqmc calculation
rhfCoeffs = numpy.eye(norb)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")

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

rhf_dipole = mf.dip_moment(unit='au')
print(f'mf dipole: {rhf_dipole}')

nuc_dipmom = [0.0, 0.0, 0.0]
for i in range(mol.natm):
  for j in range(3):
    nuc_dipmom[j] += mol.atom_charge(i)*mol.atom_coord(i)[j]

dip_ints_ao = -mol.intor_symmetric('int1e_r', comp=3)
dm1_cc = mycc.make_rdm1()
dip_ints_mo = numpy.empty_like(dip_ints_ao)
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo[i] = mf.mo_coeff.T.dot(dip_ints_ao[i]).dot(mf.mo_coeff)
edip_cc = [0., 0. ,0.]
for i in range(3):
  edip_cc[i] = numpy.trace(numpy.dot(dm1_cc, dip_ints_mo[i]))
print(f'cc dipole: {numpy.array(nuc_dipmom) + numpy.array(edip_cc)}')


print('relaxed cc dipoles:')
norbFrozen = 0
mf = scf.RHF(mol)
dE = 1.e-5
E = numpy.array([ -dE, 0., 0. ])
h1e = mf.get_hcore()
h1e += E[0] * dip_ints_ao[0]
mf.get_hcore = lambda *args: h1e
mf.verbose = 1
mf.kernel()
emf_m = mf.e_tot + E[0] * nuc_dipmom[0]

mycc = cc.CCSD(mf)
mycc.frozen = norbFrozen
mycc.verbose = 1
mycc.kernel()
emp2_m = mycc.e_hf + mycc.emp2 + E[0] * nuc_dipmom[0]
eccsd_m = mycc.e_tot + E[0] * nuc_dipmom[0]
#prepVMC.write_ccsd(mycc.t1, mycc.t2)

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
eccsdpt_m = mycc.e_tot + et + E[0] * nuc_dipmom[0]

E = numpy.array([ dE, 0., 0. ])
mf = scf.RHF(mol)
h1e = mf.get_hcore()
h1e += E[0] * dip_ints_ao[0]
mf.get_hcore = lambda *args: h1e
mf.verbose = 1
mf.kernel()
emf_p = mf.e_tot + E[0] * nuc_dipmom[0]

mycc = cc.CCSD(mf)
mycc.frozen = norbFrozen
mycc.verbose = 1
mycc.kernel()
emp2_p = mycc.e_hf + mycc.emp2 + E[0] * nuc_dipmom[0]
eccsd_p = mycc.e_tot + E[0] * nuc_dipmom[0]
#prepVMC.write_ccsd(mycc.t1, mycc.t2)

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
eccsdpt_p = mycc.e_tot + et + E[0] * nuc_dipmom[0]

print(f'emf_m: {emf_m}, emf_p: {emf_p}, dip_mf: {(emf_p - emf_m) / 2 / dE}')
print(f'emp2_m: {emp2_m}, emp2_p: {emp2_p}, dip_mp2: {(emp2_p - emp2_m) / 2 / dE}')
print(f'eccsd_m: {eccsd_m}, eccsd_p: {eccsd_p}, dip_ccsd: {(eccsd_p - eccsd_m) / 2 / dE}')
print(f'eccsdpt_m: {eccsdpt_m}, eccsd_p: {eccsdpt_p}, dip_ccsdpt: {(eccsdpt_p - eccsdpt_m) / 2 / dE}')
