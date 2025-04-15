# msc-thesis

MC-MOT

## Setup

Run the setup script to install all dependencies and environments:

```
bash setup.sh
```

This will create:
- `mvdetr_env` (Python 3.8) for detection models
- `tracking_env` (Python 3.10) for tracking algorithms

## Run an experiment

To launch an experiment with the default configuration:

```
bash scripts/run_experiment.sh --config_name default
```

Results and logs will be saved under the `experiments/` folder.
