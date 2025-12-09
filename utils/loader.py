"""
Module used to import data from the ARC dataset
"""

import json
import os

from constants import DATA
from localtypes import ColorGrid, Example, TaskData


def path_to_task(path: str) -> TaskData:
    with open(os.path.join(DATA, path), "r") as file:
        data = json.load(file)
    return data


def train_task_to_grids(
    task: str = "2dc579da.json",
) -> tuple[list[ColorGrid], list[ColorGrid], ColorGrid, ColorGrid]:
    data: TaskData = path_to_task("training/" + task)

    assert isinstance(data["train"], list), f"Error: 'train' data not a list in {task}"
    assert isinstance(data["test"], list), f"Error: 'test' data not a list in {task}"
    assert data["test"], f"Error: 'test' data is empty in {task}"

    inputs = [example["input"] for example in data["train"]]
    outputs = [example["output"] for example in data["train"]]
    input_test = data["test"][0]["input"]
    output_test = data["test"][0]["output"]

    return inputs, outputs, input_test, output_test


def train_task_to_example(task: str = "2dc579da.json", index: int = 0) -> Example:
    data: TaskData = path_to_task("training/" + task)

    example = data["train"][index]
    return example
