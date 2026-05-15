import wandb
import time
from typing import Optional


class WandbLogger:
    def __init__(
        self,
        project: str = "gomoku-transformer",
        entity: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.start_time: Optional[float] = None

        if not enabled:
            return

        wandb.init(project=project, entity=entity, config=config)

    def log(self, metrics: dict, step: int):
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_step_start(self, step: int):
        self.start_time = time.time()

    def log_step_end(self, step: int, metrics: dict):
        elapsed = time.time() - self.start_time if self.start_time else 0
        metrics["perf/step_time"] = elapsed
        self.start_time = None
        self.log(metrics, step)

    def finish(self):
        if self.enabled:
            wandb.finish()
