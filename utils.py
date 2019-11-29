class Saver(object):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save(self, path, scheduler):
        pass
