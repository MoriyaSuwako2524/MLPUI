"""Generic derivative-consistency checks for any PotentialTerm.

Use before plugging a new term into MD: forces must be -dE/dR, phi must be
dE/dQ, GESP must be d(phi)/dr, and LINEAR_CHARGE terms must satisfy
dE/dr_M = Q_M GESP_M (the assumption behind QMHub's MM gradient).
"""
from __future__ import annotations

import numpy as np

from .base import MMDependence, PotentialTerm, QMMMInput


def check_derivatives(term: PotentialTerm, inputs: QMMMInput, *, step=1e-4, n_qm=4, n_mm=4, seed=0):
    """Central finite differences on a random subset of coordinates.

    Returns {check: max abs error}; compare with the term's precision
    (float64 ~1e-7 or better for smooth models; float32 is too noisy for FD).
    """
    rng = np.random.default_rng(seed)
    available = term.available_properties()
    mm_props = [p for p in ('mm_potential', 'mm_potential_gradient') if p in available]
    ref = term.evaluate(inputs, ['energy', 'forces'] + mm_props)
    errors = {}

    def energy(x):
        return term.evaluate(x, ['energy'])['energy']

    qm_atoms = rng.choice(inputs.n_qm, size=min(n_qm, inputs.n_qm), replace=False)
    err = 0.
    for i in qm_atoms:
        for a in range(3):
            plus, minus = inputs.positions.copy(), inputs.positions.copy()
            plus[i, a] += step
            minus[i, a] -= step
            fd = -(energy(inputs.replace(positions=plus)) - energy(inputs.replace(positions=minus))) / (2 * step)
            err = max(err, abs(fd - ref['forces'][i, a]))
    errors['forces_vs_-dE/dR'] = err

    if inputs.n_mm and term.mm_dependence is not MMDependence.NONE:
        sites = rng.choice(inputs.n_mm, size=min(n_mm, inputs.n_mm), replace=False)
        if 'mm_potential' in mm_props:
            err = 0.
            for m in sites:
                plus, minus = inputs.mm_charges.copy(), inputs.mm_charges.copy()
                plus[m] += step
                minus[m] -= step
                fd = (energy(inputs.replace(mm_charges=plus)) - energy(inputs.replace(mm_charges=minus))) / (2 * step)
                err = max(err, abs(fd - ref['mm_potential'][m]))
            errors['mm_potential_vs_dE/dQ'] = err
        if 'mm_potential_gradient' in mm_props:
            # GESP is the fixed-response gradient (Q-Chem ESP_EFIELD convention): it equals
            # d(phi_M)/d(r_M) only for a probe charge, Q_M -> 0. Check it there.
            err_g, err_lin = 0., 0.
            for m in sites:
                q0 = inputs.mm_charges.copy()
                q0[m] = 0.
                probe = inputs.replace(mm_charges=q0)
                gesp0 = term.evaluate(probe, ['mm_potential_gradient'])['mm_potential_gradient'][m]
                for a in range(3):
                    plus, minus = inputs.mm_positions.copy(), inputs.mm_positions.copy()
                    plus[m, a] += step
                    minus[m, a] -= step
                    phi_p = term.evaluate(probe.replace(mm_positions=plus), ['mm_potential'])['mm_potential'][m]
                    phi_m = term.evaluate(probe.replace(mm_positions=minus), ['mm_potential'])['mm_potential'][m]
                    err_g = max(err_g, abs((phi_p - phi_m) / (2 * step) - gesp0[a]))
                    if term.mm_dependence is MMDependence.LINEAR_CHARGE:
                        xp, xm = inputs.replace(mm_positions=plus), inputs.replace(mm_positions=minus)
                        de = (energy(xp) - energy(xm)) / (2 * step)
                        err_lin = max(err_lin, abs(de - inputs.mm_charges[m] * ref['mm_potential_gradient'][m, a]))
            errors['gesp_vs_dphi/dr_probe'] = err_g
            if term.mm_dependence is MMDependence.LINEAR_CHARGE:
                errors['dE/dr_M_vs_Q*gesp'] = err_lin
    return errors
