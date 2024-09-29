import absl.app
import absl.flags


# flags
absl.flags.DEFINE_string("SENTENCE", "go to line 1", "Sentence to translate in LTL")
absl.flags.DEFINE_string(
    "LANDMARKS_FILE", "landmarks.json", "json file with landmarks"
)
absl.flags.DEFINE_string(
    "RER_PROMPT_FILE",
    "rer_prompt_augmented_2.txt",
    "txt file containing the rer prompt ",
)
absl.flags.DEFINE_string(
    "LOG_FILE",
    "results/result.csv",
    "csv file where to write the results",
)
FLAGS = absl.flags.FLAGS

from Lang2LTL.formula_sampler import ALL_PROPS

from process_extraction import sentence_2_formula


def main(argv):
    # COMPLETE EXPERIMENT
    finalLTL, _ = sentence_2_formula(
        FLAGS.SENTENCE,
        FLAGS.LANDMARKS_FILE,
        rer_model="gpt3",
        rer_engine="gpt-3.5-turbo-instruct",
        rer_prompt=FLAGS.RER_PROMPT_FILE,
        model_fpath=f"composed_model_3000000/",
        sym_trans_model="t5-base",
        convert_rule="lang2ltl",
        props=ALL_PROPS,
        log_file=FLAGS.LOG_FILE,
    )
    print(
        "--------------------------final output LTL formula---------------------------"
    )
    print(finalLTL)


if __name__ == "__main__":
    absl.app.run(main)
