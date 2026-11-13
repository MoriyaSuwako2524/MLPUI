"""Phase (sign) handling for 0 -> n transition properties.

An eigenvector is defined only up to a global phase; for real wavefunctions
that is s_n = +-1 per state, so every 0 -> n transition quantity (transition
charges, transition dipole) carries an arbitrary common sign. Labels from
different frames therefore flip at random.

The Trainer loss for one structure is

    L = w_q * mean_{i,n} (qhat_in - s_n q_in)^2 + w_mu * mean_{n,a} (muhat_na - s_n mu_na)^2

and its exact minimum over s_n in {+1, -1} is reached at

    s_n* = sign( w_q / N * qhat_n . q_n  +  w_mu / 3 * muhat_n . mu_n )

(the 1/S factor is common to both terms). Charges and dipole share s_n
because both come from the same transition density. The sign is detached:
it is a discrete label choice, not a model output.
"""
import torch


TRANSITION_TARGETS = ('transition_charges', 'transition_dipole')


def segment_sum(values, index, count):
    out = values.new_zeros((count,) + tuple(values.shape[1:]))
    return out.index_add(0, index, values)


def optimal_signs(num_states, *, q_pred=None, q_ref=None, mu_pred=None, mu_ref=None,
                  w_q=1.0, w_mu=1.0):
    """Per-state signs [S] (+1 on ties) minimising the weighted squared error.

    q_*: [N, S] (or [N] when S == 1); mu_*: [S, 3] (or [3] when S == 1).
    """
    score = None
    if q_pred is not None:
        qp, qr = q_pred.reshape(-1, num_states), q_ref.reshape(-1, num_states)
        score = w_q / max(qp.shape[0], 1) * (qp * qr).sum(0)
    if mu_pred is not None:
        mp, mr = mu_pred.reshape(num_states, 3), mu_ref.reshape(num_states, 3)
        part = w_mu / 3.0 * (mp * mr).sum(-1)
        score = part if score is None else score + part
    if score is None:
        raise ValueError('Phase alignment needs transition_charges or transition_dipole')
    score = score.detach()
    return torch.where(score < 0, -torch.ones_like(score), torch.ones_like(score))


def align_transition_targets(num_states, predictions, targets, weights):
    """Return targets with each state's sign flipped to best match the prediction."""
    present = [k for k in TRANSITION_TARGETS if k in targets]
    if not present:
        return targets
    s = optimal_signs(
        num_states,
        q_pred=predictions.get('transition_charges') if 'transition_charges' in targets else None,
        q_ref=targets.get('transition_charges'),
        mu_pred=predictions.get('transition_dipole') if 'transition_dipole' in targets else None,
        mu_ref=targets.get('transition_dipole'),
        w_q=float(weights.get('transition_charges', 0.0)),
        w_mu=float(weights.get('transition_dipole', 0.0)))
    aligned = dict(targets)
    if 'transition_charges' in targets:
        t = targets['transition_charges']
        aligned['transition_charges'] = (t.reshape(-1, num_states) * s).reshape(t.shape)
    if 'transition_dipole' in targets:
        t = targets['transition_dipole']
        aligned['transition_dipole'] = (t.reshape(num_states, 3) * s[:, None]).reshape(t.shape)
    return aligned
