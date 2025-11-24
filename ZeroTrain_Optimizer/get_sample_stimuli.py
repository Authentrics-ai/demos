import json

import huggingface_hub as hf
import pyarrow.parquet as pq


def main():
    filename = hf.hf_hub_download(
        "lextale/FirstAidInstructionsDataset",
        "data/icliniqDataset-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir="ZeroTrain_Optimizer/",
    )

    table = pq.read_table(filename)

    answers = [str(c) for c in table["answer"][:10]]

    with open("ZeroTrain_Optimizer/expected_output.txt", "w") as f:
        f.write("\n".join(answers))

    prompts = [[{"role": "user", "content": str(q)}] for q in table["question"][:10]]

    for i, p in enumerate(prompts):
        with open(f"ZeroTrain_Optimizer/prompt_{i:02d}.json", "w") as f:
            json.dump(p, f)


if __name__ == "__main__":
    main()
