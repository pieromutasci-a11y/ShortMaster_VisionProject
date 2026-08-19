import os
import argparse
from pathlib import Path

WANDB_ROOT = Path("/home/user/ros_workspace/src/vision_pipeline/models/wandb")
WANDB_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_DIR", str(WANDB_ROOT))

import yaml
import wandb

from train_sweep import main as train_and_evaluate

PROJECT_NAME = "master_detection_sweeps"
ENTITY = "pieromutasci-politecnico-di-bari"

parser = argparse.ArgumentParser()
parser.add_argument("--sweep-id", type=str, default=None,
                    help="ID di una sweep esistente da riprendere (es. sqkfh2ka)")
args = parser.parse_args()

with open(Path(__file__).parent / "sweep_config.yaml") as f:
    sweep_config = yaml.safe_load(f)

run_cap = sweep_config.pop("run_cap", 20)

if args.sweep_id:
    sweep_id = args.sweep_id
    print(f"Riprendo sweep esistente: {sweep_id}")
else:
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME, entity=ENTITY)
    print(f"Sweep creato con id: {sweep_id}")

print(f"Lancio l'agent per un massimo di {run_cap} run...")

wandb.agent(sweep_id, function=train_and_evaluate, count=run_cap, entity=ENTITY, project=PROJECT_NAME)
