import os

from openai import OpenAI


domain = []
file_path = "../config/PDDL/domain.pddl"
with open(file_path, "r") as file:
    domain = file.read()

problem = []
file_path = "../config/PDDL/problem.pddl"
with open(file_path, "r") as file:
    problem = file.read()

human_policy = []
file_path = "../config/PDDL/human_policy.pol"
with open(file_path, "r") as file:
    human_policy = file.read()

fluents = []
file_path = "../config/fluents.txt"
with open(file_path, "r") as file:
    fluents = file.read()

objects = []
file_path = "../config/objects.txt"
with open(file_path, "r") as file:
    objects = file.read()


class GPTChat:
    def __init__(self):
        self.domain = domain
        self.problem = problem
        self.human_policy = human_policy
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4-1106-preview"
        self.completion = None

    def __call__(self, psf, message):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an harvester robot in a vineyard. \
                        The vineyard is formalised in PDDL. The domain file is {self.domain}. The problem file is {self.problem}. \
                        You act according to this policy {self.human_policy}. Your goal is to describe me what you are doing \
                        depending on what state you are. The description must be provided in terms of next action to do \
                        and explanation of why you are doing it. Ignore in the state description the turndomain() and not(turndomain()) \
                        predicates. Be very short when answering. Remember to describe next actions in natural language and not with their \
                        PDDL name. Do not use technical IT terms that a farmer would not understand, like 'non-determinism'.",
                },
                {
                    "role": "user",
                    "content": "You are in the state robot-at(rob, l2), unchecked(l3), not(cleared(l3)), cleared(l2), what are you doing?",
                },
                {
                    "role": "system",
                    "content": "I have just finished to check the grape at location l2. Next I'm going to move to location l3 and check the grape there.",
                },
                {
                    "role": "user",
                    "content": "You are in the state robot-at(rob, l2), ripe(g2), empty(b), unchecked(l3), not(cleared(l3)), free(rob), what are you doing?",
                },
                {
                    "role": "system",
                    "content": "I have just found out that the grape at location l2 is ripe, and since the box is empty, I'm going to pick the grape in order to fill the box.",
                },
                {
                    "role": "user",
                    "content": f"So far you have executed the following actions of the policy: {psf}. {message}",
                },
            ],
            stream=False,
            temperature=0.0000001,
        )

        return completion.choices[0].message.content.replace("\n", "")


class FluentsExtractor:
    def __init__(self):
        self.fluents = fluents
        self.objects = objects
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4-1106-preview"
        self.completion = None

    def __call__(self, sentence):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistent which is very capable in translating natural language to PDDL fluent. \
                    You know that all the possible fluents that can occur in this domain are {fluents}, where they appear \
                    in the form of (fluent x - type_of_x y type_of_y), or (fluent x - type_of_x) or (fluent). \
                    All and only the variables admitted are the followings: {objects}. \
                    Your goal is to provide me a set of fluents used ONLY the fluents and objects that I provided you, \
                    given a sentence in a natural language. When answering, do not provide any explanation, just the set of fluents. \
                    When answering, ignore the type of the objects in the fluents.\
                    Also, provide them in the following structure '(fluent1),(fluent2),(fluent3),...' and so on. \
                    The sentence to transform is: '{sentence}' ",
                },
            ],
            stream=False,
            temperature=0.0000001,
        )

        return completion.choices[0].message.content.replace("\n", "")
