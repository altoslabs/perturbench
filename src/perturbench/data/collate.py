class noop_collate:
    """No operation collate function. Returns the batch as is."""

    def __call__(self, batch: list):
        if len(batch) == 1:
            return batch[0]
        else:
            return batch
