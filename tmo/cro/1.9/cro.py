import numpy
from pyscf import gto, scf, dft, ao2mo, mcscf, tools, lib, symm, cc
from pyscf.tools import molden
import scipy.linalg as la
from pyscf.shciscf import shci
import prepVMC
import json

df=json.load(open("/projects/anma2640/pauxy/examples/generic/tmo/trail.json"))

# Checkpoint File Name
r = 1.9
atomString = f'Cr 0 0 0; O {r} 0 0;'
el = 'Cr'
basis = 'vdz'
#mol = gto.M(atom = atomString, basis = {'Fe': 'ccpwcvdzdk', 'O': 'ccpvdzdk'}, spin = 4, symmetry = 'c2v', verbose = 4)
#mol = gto.M(atom = atomString, basis = 'ccpvdz', spin = 4, symmetry = 'c2v', verbose = 4)
#mol = gto.M(atom = atomString, basis = {'Fe': 'ano@6s5p3d2f1g', 'O': 'ano@4s3p2d1f'}, spin = 4, symmetry = 'c2v', verbose = 4)
mol = gto.Mole()
mol.ecp = {}
mol.basis = {}
for e in [el, 'O']:
  mol.ecp[e] = gto.basis.parse_ecp(df[e]['ecp'])
  mol.basis[e] = gto.basis.parse(df[e][basis])
mol.charge = 0
mol.spin = 4
mol.build(atom = atomString, verbose = 4, symmetry='c2v')
#mf = scf.fast_newton(scf.RHF(mol).x2c())
#mf = scf.ROHF(mol).x2c()
mf = scf.ROHF(mol)
mf.level_shift = 1.
mf.max_cycle = 200
#mf.irrep_nelec = { "A1": (10, 9), 'B1': (4, 3), 'B2': (4, 3), 'A2': (1, 0) }
mf.irrep_nelec = { "A1": (6, 4), 'B1': (2, 2), 'B2': (3, 2), 'A2': (1, 0) }
#mf.chkfile = 'feo.chk'
#chkfile = 'feo.chk'
#mf.__dict__.update(lib.chkfile.load(chkfile, "scf"))
mf.kernel()
mf.analyze()
tools.molden.from_mo(mol, 'cro.molden', mf.mo_coeff)
print(f'mol.nao: {mol.nao}')
print(f'mol.nelec: {mol.nelec}')
#exit(0)

norb = mol.nao
norbAct = 13
nelecAct = 10
norbFrozen = 5
ncore = 5
# active space: Cu 3d, 4d; O 2p, 3p
# O 2s optimized as core in casscf, rest core frozen at hf level
mc = shci.SHCISCF(mf,  norbAct, nelecAct)
#mo = mc.sort_mo_by_irrep({'Ag': 6,'B1u': 4,'B2u': 4, 'B3g': 4, 'B3u': 6, 'B2g': 2, 'B1g': 4, 'Au': 2},
#    {'Ag': 7,'B1u': 2,'B2u': 4, 'B3g': 0, 'B3u': 5,'B2g': 2, 'B1g': 2, 'Au': 0})
mo = mc.sort_mo_by_irrep({'A1': 5,'A2': 2,'B1': 3, 'B2': 3},
    {'A1': 3,'A2': 0,'B1': 1, 'B2': 1})
#chkfile = f'cu2o2_{f}_SHCISCF.chk'
#mc.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
mc.chkfile = f'cro_SHCISCF.chk'
#mc.frozen = norbFrozen
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-5 ]
mc.fcisolver.nPTiter = 0
mc.max_cycle_macro = 10
mc.internal_rotation = True
mc.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
mc.fcisolver.mpiprefix = "mpirun -np 36"
mc.fcisolver.scratchDirectory = "/rc_scratch/anma2640/cro/"
mc.mc1step(mo)
tools.molden.from_mo(mol, 'cro_cas.molden', mc.mo_coeff)
#exit(0)

norbAct = mol.nao
moActDice = mc.mo_coeff[:, :norbAct]
h1eff = moActDice.T.dot(mf.get_hcore()).dot(moActDice)
eri = ao2mo.kernel(mol, moActDice)
tools.fcidump.from_integrals('FCIDUMP_can_all', h1eff, eri, norbAct, mol.nelectron, mol.energy_nuc())

norbAct = 18
moActDice = mc.mo_coeff[:, :norbAct]
h1eff = moActDice.T.dot(mf.get_hcore()).dot(moActDice)
eri = ao2mo.kernel(mol, moActDice)
tools.fcidump.from_integrals('FCIDUMP_can', h1eff, eri, norbAct, mol.nelectron, mol.energy_nuc())

overlap = mf.get_ovlp(mol)
coeffs = numpy.zeros((norb, 2*norb))
coeffs[:,:norb] = prepVMC.basisChange(mf.mo_coeff, mc.mo_coeff, overlap)
coeffs[:,norb:] = prepVMC.basisChange(mf.mo_coeff, mc.mo_coeff, overlap)
prepVMC.writeMat(coeffs, 'uhf.txt')

rhfCoeffs = prepVMC.basisChange(mf.mo_coeff, mc.mo_coeff, overlap)
prepVMC.writeMat(rhfCoeffs, 'rhf.txt')

h1e, chol, nelec, enuc = prepVMC.generate_integrals(mol, mf.get_hcore(), mc.mo_coeff, chol_cut=1e-5, verbose=True)

#nbasis = h1e.shape[-1]
#rotCorevhf = moActive.T.dot(corevhf).dot(moActive)
#h1e = h1e[norbFrozen:norbFrozen+norbAct, norbFrozen:norbFrozen+norbAct] + rotCorevhf
#chol = chol.reshape((-1, nbasis, nbasis))
#chol = chol[:, norbFrozen:norbFrozen+norbAct, norbFrozen:norbFrozen+norbAct]
#mol.nelec = (mol.nelec[0]-norbFrozen, mol.nelec[1]-norbFrozen)
#enuc = energy_core

# after core averaging
nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0
chol = chol.reshape((chol.shape[0], -1))
prepVMC.write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

myucc = cc.UCCSD(mf)
myucc.verbose = 5
myucc.kernel()
#overlap = mf.get_ovlp(mol)
#rotation = (mc1.mo_coeff[:, norbFrozen:].T).dot(overlap.dot(mf.mo_coeff[:, norbFrozen:]))
#prepVMC.write_uccsd(myucc.t1, myucc.t2)

et = myucc.ccsd_t()
print('UCCSD(T) energy', myucc.e_tot + et)
