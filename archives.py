import argparse
import os
import zipfile
from typing import Dict, List, Optional

import numpy as np


def compress(directory_path: str, output_path: Optional[str] = None):

    if output_path is None:
        output_path = directory_path + ".zip"

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        controllers: Dict[str, List[str]] = {}

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.startswith("controller_") and file.endswith(".pt"):
                    if root not in controllers:
                        controllers[root] = []
                    controllers[root].append(file)
                else:
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, directory_path)
                    zipf.write(filepath, arcname)

        for root, files in controllers.items():
            numbers = [int(file.split("_")[1].split(".")[0]) for file in files]
            arg = np.argmax(numbers)
            filepath = os.path.join(root, files[arg])
            arcname = os.path.relpath(filepath, directory_path)
            zipf.write(filepath, arcname)


def extract(zip_path: str, output_path: Optional[str] = None):

    if output_path is None:
        output_path = zip_path.replace(".zip", "")

    if os.path.exists(output_path):
        raise FileExistsError(f"The directory '{output_path}' already exists.")

    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["compress", "extract"])
    parser.add_argument("path", type=str)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    assert "experiment" in args.path, "The path must be under the 'experiment' directory"

    if args.mode == "compress":
        compress(args.path, args.output)
    elif args.mode == "extract":
        extract(args.path, args.output)
    else:
        raise ValueError("Invalid mode")
