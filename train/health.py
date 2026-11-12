"""Run-local safety checks, not algorithm state or resumable schedule clocks."""


class ActorProgressGuard:
    """Stop repeated critic-only boundaries without weakening Actor KL gates.

    Counts complete eligible batches, not individual rejected candidate steps.
    A forced-only batch is not a policy stall. Explicit new runs/resumes start a
    fresh operational observation window; no optimizer/schedule state is reset.
    """

    def __init__(self, limit=3):
        if type(limit) is not int or limit <= 0:
            raise ValueError("actor_stall_limit must be a positive integer")
        self.limit = limit
        self.consecutive = 0

    def observe(self, *, actor_rows, committed_steps):
        for value in (actor_rows, committed_steps):
            if type(value) is not int or value < 0:
                raise ValueError("Actor progress counts must be nonnegative integers")
        self.consecutive = self.consecutive + 1 if actor_rows > 0 and committed_steps == 0 else 0
        return self.consecutive >= self.limit
