import os

import absl.app
import absl.flags
import openai

from process_extraction import test_on_a_set


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# flags
absl.flags.DEFINE_string(
    "DATASET",
    "dataset/DS_all_symbols.csv",
    "csv file containing sentences and target LTL formulas",
)
absl.flags.DEFINE_string(
    "LANDMARKS_FILE", "landmarks.json", "json file with landmarks"
)
absl.flags.DEFINE_string(
    "RER_PROMPT_FILE",
    "rer_prompt_augmented_2.txt",
    "txt file containing the rer prompt ",
)
absl.flags.DEFINE_string(
    "LOG_FILE", "results/results.csv", "csv file where to write the results"
)


FLAGS = absl.flags.FLAGS


def main(argv):
    test_on_a_set(
        FLAGS.DATASET, FLAGS.LANDMARKS_FILE, FLAGS.RER_PROMPT_FILE, FLAGS.LOG_FILE
    )


if __name__ == "__main__":
    absl.app.run(main)
