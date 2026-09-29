from pkg.cycle_b import Cycled


class OnCycle(Cycled):
    def __init__(self):
        self.first = self.second
        self.second = 1
