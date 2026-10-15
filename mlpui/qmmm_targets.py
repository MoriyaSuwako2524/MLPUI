"""Explicit fixed-charge baseline and train-only scaling for QMMM labels."""
import numpy as np

FOUR_TARGETS=('energy','forces','mm_potential','mm_potential_gradient')
UNITS=dict(energy='eV',forces='eV/angstrom',mm_potential='eV/e',
           mm_potential_gradient='eV/(e*angstrom)')


def fixed_charge_mm_properties(pos, mm_pos, charges, *, coulomb_constant):
    """Gas-charge potential and spatial gradient; charges already projected.

    The caller supplies the exact E/F baseline constant and charge convention.
    No charge reprojection or geometry-dependent charge response is added here.
    Positions are Angstrom, charges e, constant eV Angstrom/e^2.
    """
    p,m,q=(np.asarray(x,dtype=np.float64) for x in (pos,mm_pos,charges))
    if p.ndim!=2 or p.shape[1]!=3 or m.ndim!=2 or m.shape[1]!=3 or q.shape!=(len(p),):
        raise ValueError('Expected QM[N,3], MM[M,3], gas charges[N]')
    if not all(np.isfinite(x).all() for x in (p,m,q)) or not np.isfinite(coulomb_constant) or coulomb_constant<=0:
        raise ValueError('Coordinates, charges and Coulomb constant must be finite')
    r=p[:,None,:]-m[None,:,:]
    r2=np.sum(r*r,axis=-1)
    if np.any(r2<1e-12):
        raise ValueError('Coincident QM/MM sites are not supported')
    inv=1/np.sqrt(r2)
    phi=coulomb_constant*np.sum(q[:,None]*inv,axis=0)
    gesp=coulomb_constant*np.sum(q[:,None,None]*r*inv[:,:,None]**3,axis=0)
    return phi,gesp


def train_target_scales(train):
    """Inverse mean frame/component square, matching Trainer's loss reduction.

    Pass only the existing training split. Validation/test/guard values are
    deliberately not accepted as separate arguments or used for estimation.
    """
    sums={k:0. for k in FOUR_TARGETS};count=0
    for sample in train:
        for key in FOUR_TARGETS:
            value=np.asarray(sample[key],dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f'{key} must be finite')
            sums[key]+=float(np.mean(value*value)) if value.size else 0.
        count+=1
    if not count:
        raise ValueError('Training split must not be empty')
    means={k:v/count for k,v in sums.items()}
    if any(not np.isfinite(v) or v<=1e-24 for v in means.values()):
        raise ValueError('Target RMS too small/nonfinite for inverse-square weights')
    return dict(samples=count,scope='train only',aggregation='equal frame mean of component squares',
                units=UNITS,rms={k:float(np.sqrt(v)) for k,v in means.items()},
                loss_weights={k:1/v for k,v in means.items()})
