import json

import huggingface_hub as hf
import pyarrow.parquet as pq

SAMPLE_INDICES = [0, 2, 3, 4, 8, 9, 12, 13, 18, 20]


def main():
    filename = hf.hf_hub_download(
        "lextale/FirstAidInstructionsDataset",
        "data/icliniqDataset-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir="ZeroTrain_Optimizer_And_Maintenance/MedicalChatbot/",
    )

    table = pq.read_table(filename)

    answers = [str(table["answer"][idx]) for idx in SAMPLE_INDICES]

    with open("ZeroTrain_Optimizer_And_Maintenance/MedicalChatbot/expected_output.txt", "w") as f:
        f.write("\n".join(answers))

    prompts = [
        [{"role": "user", "content": str(table["question"][idx])}]
        for idx in SAMPLE_INDICES
    ]

    for i, p in enumerate(prompts):
        with open(f"ZeroTrain_Optimizer_And_Maintenance/MedicalChatbot/prompt_{i:02d}.json", "w") as f:
            json.dump(p, f)


if __name__ == "__main__":
    main()
