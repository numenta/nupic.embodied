# Project: emergent sparsity in unsupervised learning by disagreement


Experiments with learning by disagreement, tested in disagreement environment.

For more details please contact vclay at numenta dot com.

### Installation

Create a new environment with python 3.7:
`conda create -n <environment_name> python=3.7`
`conda activate <environment_name>`

To install nupic.embodied from source, on nupic.embodied root run: `pip install -e .`

To install disagreement specific requirements, on projects/disagreement run: `pip install -r requirements.txt`

For proper logging, make sure to set your `WANDB_API_KEY=<my_api_key>` and `WANDB_DIR=<path_to_log_directory>` environment variables. The path set in `WANDB_DIR` needs write permission for the user running the script, so make sure to set the correct permissions (for linux, use the `chmod +rwx <path_to_log_directory>` command.) See [wandbdoc](https://docs.wandb.ai/guides/track/advanced/environment-variables) for more details. If you don't have an account, create (a free) one to get an API key.

To save models, set the `CHECKPOINT_DIR=<path_to_checkpointing_directory>`. As before, mae sure to set write permission for the user running the script.

### Execution

To run an experiment, first define a new experiment in a python module under the folder experiments. Please follow the example of other configs already created.

To run, on projects/disagreement call `python run.py -e <experiment_name>`
