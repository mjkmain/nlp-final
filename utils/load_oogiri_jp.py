# Oogiri dataset for cold-start SFT, will be translated into Korean using GPT API.

import os
import datasets

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_oogiri():
    save_path = os.path.join(root, "src", "data", "oogiri_cold_start_jp")

    if os.path.exists(save_path):
        return datasets.load_from_disk(save_path)

    ds = datasets.load_dataset(
        "Joctor/bokete_oogiri_caption",
        name="default",
        split="train"
    )

    ds = ds.sort(column_names="star", reverse=True)
    ds = ds.select(range(200_000))
    ds.save_to_disk(save_path, num_proc=48)

    return ds

if __name__ == "__main__":
    load_oogiri()
