class CheckSize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input):
        assert input.size == (self.size[1], self.size[0])

        return input
