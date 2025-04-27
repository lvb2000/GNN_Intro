from dataclasses import dataclass

@dataclass
class ExtendedSchedulerConfig:
    scheduler: str = 'cosine_with_warmup'
    max_epoch: int = 200
    num_warmup_epochs: int = 10
    train_mode: str = 'custom'
    eval_period: int = 1
    reduce_factor: float = 0.5
    schedule_patience: int = 15
    min_lr: float = 1e-6
