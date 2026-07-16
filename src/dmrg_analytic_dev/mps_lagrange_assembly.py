"""MPS-aware Lagrange nuclear-gradient assembly helpers.

PySCF's SA-CASSCF gradient/NAC assembly accepts a dense CI Lagrange vector and
calls ``mc.fcisolver.trans_rdm12(Lci, ci, ...)`` internally.  For a large active
space, the response vector may only exist as an MPS.  The functions here split
that dependency: callers provide the transition 1/2-RDMs directly, and the
remaining AO derivative contractions follow PySCF's ``Lci_dot_dgci_dx`` algebra.

This is a narrow replacement for the CI-Lagrange contribution only.  The
orbital-Lagrange contribution and Hamiltonian-response terms can still use the
existing PySCF code, provided their required state RDMs are supplied by the DMRG
solver.
"""

from __future__ import annotations

from functools import reduce

import numpy as np
from pyscf import ao2mo, lib
from pyscf.grad.mp2 import _shell_prange
from pyscf.lib import logger


def Lorb_dot_dgorb_dx_from_rdms(
    Lorb,
    casdm1,
    casdm2,
    mc,
    mo_coeff=None,
    atmlst=None,
    mf_grad=None,
    eris=None,
    verbose=None,
):
    """Orbital-Lagrange nuclear derivative from active-space SA RDMs.

    This is the RDM-driven analogue of
    ``pyscf.grad.sacasscf.Lorb_dot_dgorb_dx``.  It avoids the dense-CI call
    ``mc.fcisolver.make_rdm12(ci, ...)`` by accepting the state-averaged active
    1/2-RDMs directly.
    """
    t0 = (logger.process_clock(), logger.perf_counter())
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if mf_grad is None:
        mf_grad = mc._scf.nuc_grad_method()
    if eris is None:
        eris = mc.ao2mo(mo_coeff)
    if mc.frozen is not None:
        raise NotImplementedError

    Lorb = np.asarray(Lorb)
    casdm1 = np.asarray(casdm1)
    casdm2 = np.asarray(casdm2)
    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao + 1) // 2

    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]
    moL_coeff = np.dot(mo_coeff, Lorb)
    s0_inv = np.dot(mo_coeff, mo_coeff.T)
    moL_core = moL_coeff[:, :ncore]
    moL_cas = moL_coeff[:, ncore:nocc]

    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    dmL_core = np.dot(moL_core, mo_core.T) * 2
    dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
    dmL_core += dmL_core.T
    dmL_cas += dmL_cas.T
    dm1 = dm_core + dm_cas
    dm1L = dmL_core + dmL_cas

    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=dm_cas.dtype)
    aapaL = np.zeros((ncas, ncas, nmo, ncas), dtype=dm_cas.dtype)
    for i in range(nmo):
        jbuf = eris.ppaa[i]
        kbuf = eris.papa[i]
        aapa[:, :, i, :] = jbuf[ncore:nocc, :, :].transpose(1, 2, 0)
        aapaL[:, :, i, :] += np.tensordot(
            jbuf, Lorb[:, ncore:nocc], axes=((0), (0)),
        )
        kbuf_l = np.tensordot(
            kbuf, Lorb[:, ncore:nocc], axes=((1), (0)),
        ).transpose(1, 2, 0)
        aapaL[:, :, i, :] += kbuf_l + kbuf_l.transpose(1, 0, 2)

    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    vjL, vkL = mc._scf.get_jk(mol, (dmL_core, dmL_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * 0.5
    vhf_a = vj[1] - vk[1] * 0.5
    vhfL_c = vjL[0] - vkL[0] * 0.5
    vhfL_a = vjL[1] - vkL[1] * 0.5
    gfock = np.dot(h1, dm1L)
    gfock += np.dot((vhf_c + vhf_a), dmL_core)
    gfock += np.dot((vhfL_c + vhfL_a), dm_core)
    gfock += np.dot(vhfL_c, dm_cas)
    gfock += np.dot(vhf_c, dmL_cas)
    gfock = np.dot(s0_inv, gfock)
    gfock += reduce(
        np.dot,
        (mo_coeff, np.einsum("uviw,uvtw->it", aapaL, casdm2), mo_cas.T),
    )
    gfock += reduce(
        np.dot,
        (mo_coeff, np.einsum("uviw,vuwt->it", aapa, casdm2), moL_cas.T),
    )
    dme0 = (gfock + gfock.T) / 2
    aapaL = vj = vk = vhf_c = vhf_a = None

    vj, vk = mf_grad.get_jk(mol, (dm_core, dm_cas, dmL_core, dmL_cas))
    vhf1c, vhf1a, vhf1cL, vhf1aL = vj - vk * 0.5
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    diag_idx = np.arange(nao)
    diag_idx = diag_idx * (diag_idx + 1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0, 1, 3, 2)
    dm2buf = ao2mo._ao2mo.nr_e2(
        casdm2_cc.reshape(ncas**2, ncas**2),
        mo_cas.T,
        (0, nao, 0, nao),
    ).reshape(ncas**2, nao, nao)

    dm2Lbuf = np.zeros((ncas**2, nmo, nmo))
    Lcasdm2 = np.tensordot(
        Lorb[:, ncore:nocc], casdm2, axes=(1, 2),
    ).transpose(1, 2, 0, 3)
    dm2Lbuf[:, :, ncore:nocc] = Lcasdm2.reshape(ncas**2, nmo, ncas)
    Lcasdm2 = np.tensordot(
        Lorb[:, ncore:nocc], casdm2, axes=(1, 3),
    ).transpose(1, 2, 3, 0)
    dm2Lbuf[:, ncore:nocc, :] += Lcasdm2.reshape(ncas**2, ncas, nmo)
    dm2Lbuf += dm2Lbuf.transpose(0, 2, 1)
    dm2Lbuf = np.ascontiguousarray(dm2Lbuf)
    dm2Lbuf = ao2mo._ao2mo.nr_e2(
        dm2Lbuf.reshape(ncas**2, nmo**2),
        mo_coeff.T,
        (0, nao, 0, nao),
    ).reshape(ncas**2, nao, nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:, diag_idx] *= 0.5
    dm2buf = dm2buf.reshape(ncas, ncas, nao_pair)
    dm2Lbuf = lib.pack_tril(dm2Lbuf)
    dm2Lbuf[:, diag_idx] *= 0.5
    dm2Lbuf = dm2Lbuf.reshape(ncas, ncas, nao_pair)

    if atmlst is None:
        atmlst = list(range(mol.natm))
    atmlst = list(atmlst)
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst), 3))
    de_renorm = np.zeros((len(atmlst), 3))
    de_eri = np.zeros((len(atmlst), 3))

    max_memory = mc.max_memory - lib.current_memory()[0]
    blksize = int(
        max_memory * 0.9e6 / 8
        / (4 * (aoslices[:, 3] - aoslices[:, 2]).max() * nao_pair)
    )
    blksize = min(nao, max(2, blksize))
    log = logger.new_logger(mc, verbose)
    log.info(
        "SA-CASSCF MPS Lorb_dot_dgorb memory remaining for eri manipulation: "
        "%f MB; using blocksize = %d",
        max_memory, blksize,
    )
    t0 = log.timer("SA-CASSCF MPS Lorb_dot_dgorb 1-electron part", *t0)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de_hcore[k] += np.einsum("xij,ij->x", h1ao, dm1L)
        de_renorm[k] -= np.einsum("xij,ij->x", s1[:, p0:p1], dme0[p0:p1]) * 2

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum(
                "ijw,pi,qj->pqw",
                dm2Lbuf,
                mo_cas[p0:p1],
                mo_cas[q0:q1],
            )
            dm2_ao += lib.einsum(
                "ijw,pi,qj->pqw",
                dm2buf,
                moL_cas[p0:p1],
                mo_cas[q0:q1],
            )
            dm2_ao += lib.einsum(
                "ijw,pi,qj->pqw",
                dm2buf,
                mo_cas[p0:p1],
                moL_cas[q0:q1],
            )
            shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor(
                "int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice,
            ).reshape(3, p1 - p0, nf, nao_pair)
            de_eri[k] -= np.einsum("xijw,ijw->x", eri1, dm2_ao) * 2
            t0 = log.timer(
                "SA-CASSCF MPS Lorb_dot_dgorb atom {} ({},{}|{})".format(
                    ia, p1 - p0, nf, nao_pair,
                ),
                *t0,
            )
        de_eri[k] += np.einsum("xij,ij->x", vhf1c[:, p0:p1], dm1L[p0:p1]) * 2
        de_eri[k] += np.einsum("xij,ij->x", vhf1cL[:, p0:p1], dm1[p0:p1]) * 2
        de_eri[k] += np.einsum("xij,ij->x", vhf1a[:, p0:p1], dmL_core[p0:p1]) * 2
        de_eri[k] += np.einsum("xij,ij->x", vhf1aL[:, p0:p1], dm_core[p0:p1]) * 2

    return de_hcore + de_renorm + de_eri


def Lci_dot_dgci_dx_from_tdm(
    casdm1,
    casdm2,
    mc,
    mo_coeff=None,
    atmlst=None,
    mf_grad=None,
    eris=None,
    verbose=None,
    *,
    symmetrize: bool = True,
):
    """CI-Lagrange nuclear derivative from active-space transition RDMs.

    Parameters
    ----------
    casdm1, casdm2
        Active-space transition RDMs corresponding to the weighted SA sum
        ``sum_I w_I <L_I|E|I>``.  If ``symmetrize=True`` (default), this
        function applies the same ``+ transpose`` symmetrization as PySCF's
        ``grad.sacasscf.Lci_dot_dgci_dx``.
    mc
        SA-CASSCF object.

    Returns
    -------
    de : ndarray, shape ``(len(atmlst), 3)``
        CI-Lagrange contribution to the nuclear derivative.
    """
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if mf_grad is None:
        mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError
    if eris is None:
        eris = mc.ao2mo(mo_coeff)

    casdm1 = np.asarray(casdm1).copy()
    casdm2 = np.asarray(casdm2).copy()
    if symmetrize:
        casdm1 = casdm1 + casdm1.T
        casdm2 = casdm2 + casdm2.transpose(1, 0, 3, 2)

    t0 = (logger.process_clock(), logger.perf_counter())
    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao + 1) // 2

    mo_occ = mo_coeff[:, :nocc]
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]

    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=dm_cas.dtype)
    for i in range(nmo):
        aapa[:, :, i, :] = eris.ppaa[i][ncore:nocc, :, :].transpose(1, 2, 0)
    vj, vk = mc._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = mc.get_hcore()
    vhf_c = vj[0] - vk[0] * 0.5
    vhf_a = vj[1] - vk[1] * 0.5
    gfock = np.zeros_like(dm_cas)
    gfock[:, :nocc] = reduce(np.dot, (mo_coeff.T, vhf_a, mo_occ)) * 2
    gfock[:, ncore:nocc] = reduce(
        np.dot, (mo_coeff.T, h1 + vhf_c, mo_cas, casdm1),
    )
    gfock[:, ncore:nocc] += np.einsum("uvpw,vuwt->pt", aapa, casdm2)
    dme0 = reduce(np.dot, (mo_coeff, (gfock + gfock.T) * 0.5, mo_coeff.T))
    aapa = vj = vk = vhf_c = vhf_a = h1 = gfock = None

    vj, vk = mf_grad.get_jk(mol, (dm_core, dm_cas))
    vhf1c, vhf1a = vj - vk * 0.5
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)

    diag_idx = np.arange(nao)
    diag_idx = diag_idx * (diag_idx + 1) // 2 + diag_idx
    casdm2_cc = casdm2 + casdm2.transpose(0, 1, 3, 2)
    dm2buf = ao2mo._ao2mo.nr_e2(
        casdm2_cc.reshape(ncas**2, ncas**2),
        mo_cas.T,
        (0, nao, 0, nao),
    ).reshape(ncas**2, nao, nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:, diag_idx] *= 0.5
    dm2buf = dm2buf.reshape(ncas, ncas, nao_pair)
    casdm2 = casdm2_cc = None

    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = list(atmlst)
    aoslices = mol.aoslice_by_atom()
    de_hcore = np.zeros((len(atmlst), 3))
    de_renorm = np.zeros((len(atmlst), 3))
    de_eri = np.zeros((len(atmlst), 3))

    max_memory = mc.max_memory - lib.current_memory()[0]
    blksize = int(
        max_memory * 0.9e6 / 8
        / (4 * (aoslices[:, 3] - aoslices[:, 2]).max() * nao_pair)
    )
    blksize = min(nao, max(2, blksize))
    log = logger.new_logger(mc, verbose)
    log.info(
        "SA-CASSCF MPS Lci_dot_dgci memory remaining for eri manipulation: "
        "%f MB; using blocksize = %d",
        max_memory, blksize,
    )
    t0 = log.timer("SA-CASSCF MPS Lci_dot_dgci 1-electron part", *t0)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de_hcore[k] += np.einsum("xij,ij->x", h1ao, dm_cas)
        de_renorm[k] -= np.einsum("xij,ij->x", s1[:, p0:p1], dme0[p0:p1]) * 2

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum(
                "ijw,pi,qj->pqw",
                dm2buf,
                mo_cas[p0:p1],
                mo_cas[q0:q1],
            )
            shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
            eri1 = mol.intor(
                "int2e_ip1", comp=3, aosym="s2kl", shls_slice=shls_slice,
            ).reshape(3, p1 - p0, nf, nao_pair)
            de_eri[k] -= np.einsum("xijw,ijw->x", eri1, dm2_ao) * 2
            t0 = log.timer(
                "SA-CASSCF MPS Lci_dot_dgci atom {} ({},{}|{})".format(
                    ia, p1 - p0, nf, nao_pair,
                ),
                *t0,
            )
        de_eri[k] += np.einsum("xij,ij->x", vhf1c[:, p0:p1], dm_cas[p0:p1]) * 2
        de_eri[k] += np.einsum("xij,ij->x", vhf1a[:, p0:p1], dm_core[p0:p1]) * 2

    return de_hcore + de_renorm + de_eri
