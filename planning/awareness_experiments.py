import json
import os
import random

import tyro
from GPT_Agents import FluentsExtractor
from process_states import (
    evaluate_metric,
    get_current_state,
    get_future_states,
    get_past_states,
    load_plan,
    simulate_plan,
)


PLAN_PATH = "../config/PDDL/sas_plan_adapted"

questions_first_set = [
    "What are you doing now?",
    "What's occupying your time at the moment?",
    "What's your current activity?",
    "What are you currently engaged in?",
    "What's keeping you busy right now?",
    "What task are you working on now?",
]

questions_second_set = [
    "What did you do so far?",
    "What have you accomplished so far?",
    "What tasks have you completed up to this point?",
    "What progress have you made until now?",
    "What have you handled so far?",
    "What have you achieved up to this moment?",
]

questions_third_set = [
    "What are you going to do next?",
    "What's your next move?",
    "What do you plan to do afterward?",
    "What's up next on your to-do list?",
    "What's your next step going to be?",
    "What's the next course of action for you?",
]

questions_dict = {
    "Current_action": questions_first_set,
    "Past_actions": questions_second_set,
    "Future_actions": questions_third_set,
}


def main(n_exp: int = 15) -> None:
    extractor = FluentsExtractor()

    plan = load_plan(PLAN_PATH)

    for category, questions in questions_dict.items():
        print(f"Category: {category}")

        output_dir_ = f"results/{category}"

        if not os.path.exists(output_dir_):
            os.makedirs(output_dir_)

        for i in range(n_exp):
            print(f"Experiment {i+1}")
            question = random.choice(questions)
            system_response, psf_returned = simulate_plan(plan, question, 0.25)
            output_dir = f"{output_dir_}/experiment_{i+1}"

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            extractor = FluentsExtractor()
            fluents = extractor(system_response)
            fluents = fluents.split(",")

            meaningful_states = None

            if category == "Current_action":
                meaningful_states = get_current_state(psf_returned)
            elif category == "Past_actions":
                meaningful_states = get_past_states(psf_returned)
            elif category == "Future_actions":
                meaningful_states = get_future_states(psf_returned, plan)

            hits = evaluate_metric(meaningful_states, fluents)

            results = {}
            results["user_question"] = question
            results["system_response"] = system_response
            results["extracted_fluents"] = fluents
            results["hits"] = hits

            output_file = f"{output_dir}/output.json"
            with open(output_file, "w") as file:
                json.dump(results, file, indent=4)

        print("--------------------------------------------------------")

    return


if __name__ == "__main__":
    tyro.cli(main)
