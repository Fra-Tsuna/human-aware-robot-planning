import json
import os

import matplotlib.pyplot as plt

import numpy as np

categories = [
    "Current_action",
    "Past_actions",
    "Future_actions",
]


def main(n_exp: int = 30) -> None:
    for category in categories:
        result_dir = f"results/{category}"

        soundness_list = []
        completeness_list = []
        for i in range(n_exp):
            output_file = f"{result_dir}/experiment_{i+1}/output.json"

            with open(output_file, "r") as file:
                results = json.load(file)

            soundness = results["soundness"]
            completeness = results["completeness"]
            soundness_list.append(soundness)
            completeness_list.append(completeness)
            
        completeness_average = np.mean(completeness_list)
        completeness_std = np.std(completeness_list)
        soundness_average = np.mean(soundness_list)
        soundness_std = np.std(soundness_list)

        print(f"Category: {category}\n")
        print(f"Completeness statistics: {completeness_average:.2f} +- {completeness_std:.2f}\n")
        print(f"Soundness statistics: {soundness_average:.2f} +- {soundness_std:.2f}\n")


if __name__ == "__main__":
    main()
