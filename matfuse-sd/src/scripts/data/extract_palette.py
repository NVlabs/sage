import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from Pylette import extract_colors
from tqdm import tqdm


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("--data", help = "Dataset folder.")

    # Read arguments from command line
    args = parser.parse_args()

    data_dir = Path(args.data).expanduser()
    for item_dir in tqdm(sorted(data_dir.glob("**/renders/"))):
        mat_dir = item_dir.parent

        render = list(item_dir.glob("*.png"))[0]

        palette = extract_colors(str(render), palette_size=5, sort_mode="frequency")
        colors = [c.rgb for c in palette]

        metadata = json.load(open(mat_dir / "metadata.json"))
        metadata["palette"] = colors

        with open(mat_dir / "metadata.json", "w") as f:
            json.dump(colors, f, cls=NpEncoder, indent=4)
