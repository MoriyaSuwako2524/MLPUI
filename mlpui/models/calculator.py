
import ase


class BaseCalculator(ase.calculators.calculator.Calculator):
    def __init__(self):
        super().__init__()
