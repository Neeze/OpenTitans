class Trainer:
    """Trainer handling Backpropagation Through Time (BPTT)."""
    def __init__(self, model, args, train_dataset):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset

    def train(self):
        pass
