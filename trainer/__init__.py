"""bittle-trainer: RL locomotion training service for the Petoi Bittle X.

Runs on a GPU box (not in the bittle-agent NAS container). bittle-agent talks
to it over HTTP (see ``trainer/service.py``); the LLM only ever edits a
validated ``TrainConfig`` and reads gate reports.
"""

__version__ = "0.1.0"
