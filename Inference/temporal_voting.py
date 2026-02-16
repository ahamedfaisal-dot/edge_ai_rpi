from collections import deque

class MultiFrameVoter:
    def __init__(self, window_size=3, vote_threshold=2):
        self.window_size = window_size
        self.vote_threshold = vote_threshold
        self.history = deque(maxlen=window_size)

    def update(self, detected: bool):
        self.history.append(1 if detected else 0)

        if len(self.history) < self.window_size:
            return False  # not enough frames yet

        return sum(self.history) >= self.vote_threshold
