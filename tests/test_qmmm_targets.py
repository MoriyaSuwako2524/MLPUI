import numpy as np
import pytest


def test_fixed_charge_gradient_sign_units_and_fd():
    from mlpui.qmmm_targets import fixed_charge_mm_properties
    p=np.array([[0.,0.,0.]])
    m=np.array([[2.,0.,0.],[0.,3.,0.]])
    q=np.array([1.])
    phi,g=fixed_charge_mm_properties(p,m,q,coulomb_constant=2.)
    np.testing.assert_allclose(phi,[1.,2/3])
    np.testing.assert_allclose(g,[[-.5,0,0],[0,-2/9,0]])
    for axis in range(3):
        shift=np.zeros_like(m);shift[:,axis]=1e-5
        plus,_=fixed_charge_mm_properties(p,m+shift,q,coulomb_constant=2.)
        minus,_=fixed_charge_mm_properties(p,m-shift,q,coulomb_constant=2.)
        np.testing.assert_allclose((plus-minus)/2e-5,g[:,axis],atol=1e-10)
    with pytest.raises(ValueError,match='Coincident'):
        fixed_charge_mm_properties(p,p,q,coulomb_constant=2.)


def test_train_scale_equal_frame_ragged_and_excludes_other_splits():
    from mlpui.qmmm_targets import train_target_scales
    data=[{'energy':np.array(2.),'forces':np.ones((2,3))*3,
           'mm_potential':np.ones(1)*4,'mm_potential_gradient':np.ones((1,3))*5},
          {'energy':np.array(4.),'forces':np.ones((2,3))*6,
           'mm_potential':np.ones(9)*8,'mm_potential_gradient':np.ones((9,3))*10}]
    result=train_target_scales(data)
    assert result['samples']==2
    for key,rms2 in zip(data[0],[10,22.5,40,62.5]):
        assert result['rms'][key] == pytest.approx(np.sqrt(rms2))
        assert result['loss_weights'][key] == pytest.approx(1/rms2)
    data[0]['energy']=np.nan
    with pytest.raises(ValueError,match='finite'):
        train_target_scales(data)
