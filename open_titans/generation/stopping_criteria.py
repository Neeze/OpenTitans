class StoppingCriteria:
    """Criteria to stop token generation."""
    def __call__(self, input_ids, scores, **kwargs):
        return False
