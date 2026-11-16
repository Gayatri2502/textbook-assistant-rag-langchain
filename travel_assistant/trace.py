"""Structured, request-scoped events for the pipeline explorer."""
from time import perf_counter


class Trace:
    def __init__(self):
        self.started = perf_counter()

    def event(self, stage, state, *, duration=None, **detail):
        event = {
            "type": "stage", "stage": stage, "state": state,
            "elapsed_ms": round((perf_counter() - self.started) * 1000, 2),
            "detail": detail,
        }
        if duration is not None:
            event["duration_ms"] = round(duration * 1000, 2)
        return event

    def step(self, stage, operation, describe=lambda _: {}, **input_detail):
        yield self.event(stage, "running", **input_detail)
        started = perf_counter()
        try:
            value = operation()
            detail = describe(value)
        except Exception:
            yield self.event(stage, "error", duration=perf_counter() - started,
                             message="This step failed. Check the service configuration and try again.")
            raise
        yield self.event(stage, "done", duration=perf_counter() - started, **input_detail, **detail)
        return value
