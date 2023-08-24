import click

from step_01_process_embeddings import main as step_01
from step_02_train_apply_pca import main as step_02
from step_03_bootstrap_milvus import main as step_03


@click.command()
@click.option("--overwrite-collection", is_flag=True, default=False, help="Overwrite collection if it exists")
@click.option("--steps", default="all", help="Comma separated list of steps to run")
@click.option("--load-percentage", default=25, help="Percentage of data to load", type=float)
def run_pipeline(overwrite_collection, steps, load_percentage):
    allowed_steps = ("all", "2_and_3", "only_3")
    if steps not in allowed_steps:
        raise ValueError(f"Invalid steps argument: {steps}. Must be one of {allowed_steps}")
    if steps == "all":
        step_01()
        step_02()
        step_03(overwrite_collection, load_percentage)
    elif steps == "2_and_3":
        step_02()
        step_03(overwrite_collection, load_percentage)
    elif steps == "only_3":
        step_03(overwrite_collection, load_percentage)


if __name__ == "__main__":
    run_pipeline()
