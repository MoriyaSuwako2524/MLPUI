"""Sum of potential terms; itself a PotentialTerm, so composites nest."""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .base import ADDITIVE_PROPERTIES, MMDependence, PotentialTerm, QMMMInput


class CompositePotential(PotentialTerm):
    """E = sum_k E_k, and likewise for every additive derivative property.

    After each ``evaluate`` the per-term results are in ``self.components``
    (term name -> property dict) and per-term diagnostics in ``self.extras``
    (keys ``"<term>:<key>"``), e.g. for monitoring during MD.
    """

    def __init__(self, terms: Sequence[PotentialTerm], name='composite'):
        super().__init__(name)
        terms = list(terms)
        if not terms or not all(isinstance(t, PotentialTerm) for t in terms):
            raise ValueError('CompositePotential needs at least one PotentialTerm')
        names = [t.name for t in terms]
        if len(set(names)) != len(names):
            raise ValueError(f'duplicate term names: {names}')
        self.terms = terms
        self.mm_dependence = MMDependence(max(t.mm_dependence for t in terms))
        common = set(ADDITIVE_PROPERTIES)
        for term in terms:
            common &= term.available_properties()
        self.properties = frozenset(common)
        self.components: dict[str, dict] = {}

    def __getitem__(self, name):
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(name)

    def _compute(self, inputs: QMMMInput, properties):
        total, components, extras = {}, {}, {}
        for term in self.terms:
            result = term.evaluate(inputs, properties)
            components[term.name] = result
            extras.update({f'{term.name}:{k}': v for k, v in term.extras.items()})
            for prop, value in result.items():
                total[prop] = total[prop] + value if prop in total else np.array(value, dtype=np.float64)
        self.components = components
        self.extras = extras
        return total

    def metadata(self):
        return {**super().metadata(), 'terms': [t.metadata() for t in self.terms]}

    def describe(self):
        return json.dumps(self.metadata(), indent=1, default=str)
