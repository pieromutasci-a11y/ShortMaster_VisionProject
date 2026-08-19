import wandb

api = wandb.Api()
sweep = api.sweep("pieromutasci-politecnico-di-bari/master_detection_sweeps/sqkfh2ka")
sweep.config["program"] = "run_sweep.py"
sweep.config["command"] = ["python3", "${program}", "${args}"]
sweep.update()
print("Done")
