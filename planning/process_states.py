import json
import random
from typing import Union, List

from GPT_Agents import *


INIT_STATE = [
    "(robot-at rob l0)",
    "(robot-at support0 l0)",
    "(support support0)",
    "(logistic rob)",
    "(grape-at g0 l0)",
    "(grape-at g1 l1)",
    "(grape-at g2 l2)",
    "(grape-at g3 l3)",
    "(unchecked l0)",
    "(unchecked l1)",
    "(unchecked l2)",
    "(unchecked l3)",
    "(free rob)",
    "(in b rob)",
    "(adj l0 l1)",
    "(adj l1 l2)",
    "(adj l2 l3)",
    "(full b)",
]

PLAN_PATH = "config/PDDL/sas_plan_adapted"

domain = []
DOMAIN_PATH = "config/PDDL/domain.pddl"
with open(DOMAIN_PATH, "r") as file:
    domain = file.read()

problem = []
PROBLEM_PATH = "config/PDDL/problem.pddl"
with open(PROBLEM_PATH, "r") as file:
    problem = file.read()

human_policy = []
POLICY_PATH = "config/PDDL/human_policy.pol"
with open(POLICY_PATH, "r") as file:
    human_policy = file.read()

chat_kwargs = {"domain": domain, "problem": problem, "human_policy": human_policy}

with open("config/action_schemas.json", "r") as file:
    ACTIONS_SCHEMA = json.load(file)


def process_action(action):
    # split name and arguments
    action = action.replace("(", "").replace(")", "")
    action = action.split(" ")
    action = {
        "name": action[0],
        "args": action[1:],
    }

    # print("ACTION ", action)

    action_schema = ACTIONS_SCHEMA[action["name"]]
    add_set = action_schema["add_set"].split(",")
    del_set = action_schema["del_set"].split(",")
    for i, arg in enumerate(action["args"]):
        add_set = [x.replace(f"?{i+1}", arg) for x in add_set]
        del_set = [x.replace(f"?{i+1}", arg) for x in del_set]

    return action, set(add_set), set(del_set)


def add_fluents(state, add_set):
    for fluent in add_set:
        if fluent not in state:
            state.add(fluent)
    return state


def remove_fluents(state, del_set):
    for fluent in del_set:
        if fluent in state:
            state.remove(fluent)
    return state


def get_next_state(state, action):
    action, add_set, del_set = process_action(action)

    state = add_fluents(state, add_set)
    state = remove_fluents(state, del_set)

    return state


def get_current_state(plan_so_far, gt_plan):
    state = INIT_STATE.copy()
    state = set(state)
    for i in range(len(plan_so_far)):
        action = plan_so_far[i]
        if "check_grape" in action:
            next_action = gt_plan[i + 1]
            if "handle_exception" in next_action:
                action = action.replace("check_grape", "check_grape_uk")
            elif "assest_vine" in next_action:
                action = action.replace("check_grape", "check_grape_ur")
            else:
                action = action.replace("check_grape", "check_grape_rp")
        state = get_next_state(state, action)

    return set(state)


def evaluate_metric(plan_so_far_returned, extracted_fluents, category):
    gamma = 0.0
    gammas = []
    gt_states = []
    soundness_ = []
    correct_list = []
    missing_list = []
    hallucination_list = []
    union_list = []
    histogram = []
    gt_plan = load_plan(PLAN_PATH)

    if category == "Current_action":
        # Equation 1 in the paper
        real_states = get_current_state(plan_so_far_returned, gt_plan)
        intersection = real_states & extracted_fluents
        gt_states.append(list(real_states))
        gamma = len(intersection) / len(real_states)
        soundness_.append(len(intersection) / len(extracted_fluents))
        soundness = sum(soundness_) / len(soundness_)
        gammas.append(gamma)

    elif category == "Past_actions":
        # Equation 2 in the paper
        cumulative_plan = []
        gamma_past = 0.0
        t = 0
        for action in plan_so_far_returned:
            cumulative_plan.append(action)
            real_states = get_current_state(cumulative_plan, gt_plan)
            gt_states.append(list(real_states))
            intersection = real_states & extracted_fluents
            gamma_ = len(intersection) / len(real_states)
            gamma_past += gamma_
            gammas.append(gamma_)
            soundness_.append(len(intersection) / len(extracted_fluents))
            correct_list.append(len(intersection))
            missing_list.append(len(real_states.difference(extracted_fluents)))
            hallucination_list.append(len(extracted_fluents.difference(real_states)))
            union_list.append(len(real_states.union(extracted_fluents)))
            t += 1
        gamma = gamma_past / t
        histogram = {
            "correct": correct_list,
            "missing": missing_list,
            "hallucinations": hallucination_list,
            "union": union_list,
            "zero": len(plan_so_far_returned),
        }
        soundness = sum(soundness_) / len(soundness_)

    elif category == "Future_actions":
        # Equation 3 in the paper
        t = len(plan_so_far_returned)
        cumulative_plan = plan_so_far_returned[:]
        gamma_future = 0.0
        i = 0
        for action in gt_plan[t:]:
            real_states = get_current_state(cumulative_plan, gt_plan)
            gt_states.append(list(real_states))
            intersection = real_states & extracted_fluents
            gamma_ = len(intersection) / len(real_states)
            gammas.append(gamma_)
            gamma_future += gamma_
            soundness_.append(len(intersection) / len(extracted_fluents))
            correct_list.append(len(intersection))
            missing_list.append(len(real_states.difference(extracted_fluents)))
            hallucination_list.append(len(extracted_fluents.difference(real_states)))
            union_list.append(len(real_states.union(extracted_fluents)))
            cumulative_plan.append(action)
            i += 1
        gamma = gamma_future / i
        histogram = {
            "correct": correct_list,
            "missing": missing_list,
            "hallucinations": hallucination_list,
            "union": union_list,
            "zero": len(plan_so_far_returned),
        }
        soundness = sum(soundness_) / len(soundness_)
    else:
        raise ValueError("Invalid category")

    return gamma, gammas, soundness, histogram, gt_states


def simulate_plan(plan, question, question_probability=0.25, baseline=False):
    p = question_probability
    increment = (1 - p) / len(plan)
    plan_so_far = []
    psf_returned = []
    system_response = []
    for i, action in enumerate(plan):
        plan_so_far.append(f"{i+1}) {action}")
        psf_returned.append(action)

        if random.random() < p:
            user_response = question
            if not baseline:
                chat = GPTChat(**chat_kwargs)
                system_response = chat(plan_so_far, user_response)
            else:
                chat = NaiveBaseline()
                system_response = chat(plan_so_far, user_response)
            break
        else:
            p += increment
            p = min(1, p)

    return system_response, psf_returned


def load_plan(plan_path):
    plan = []
    with open(plan_path, "r") as file:
        temp = file.readlines()
        plan.extend(line.replace("\n", "") for line in temp)
    return plan
