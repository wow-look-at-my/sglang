from torch.nn import Conv2d


class Base:
    def __init__(self):
        self.reset()

    def reset(self):
        pass


class EagerBase(Conv2d):
    pass
