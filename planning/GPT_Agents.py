import os
from openai import OpenAI

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

class GPTChat:
    def __init__(self, **kwargs):
        self.domain = kwargs["domain"]
        self.problem = kwargs["problem"]
        self.human_policy = kwargs["human_policy"]
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4o"
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
                    "content": "I have just finished to check the grape at location l2. After handling this grape, I am going to move to location l3 and check the grape there.",
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
            temperature=1e-7,
        )

        return completion.choices[0].message.content.replace("\n", "")


class NaiveBaseline:
    def __init__(self, **kwargs):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4o"
        self.completion = None

    def __call__(self, psf, message):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an harvester robot in a vineyard.\
                        Your goal is to describe me what you are doing \
                        depending on what state you are. The description must be provided in terms of next action to do \
                        and explanation of why you are doing it. Be very short when answering. Remember to describe next actions in natural language and not with their \
                        PDDL name. Do not use technical IT terms that a farmer would not understand, like 'non-determinism'.",
                },
                {
                    "role": "user",
                    "content": f"So far you have executed the following actions of the policy: {psf}. {message}",
                },
            ],
            stream=False,
            temperature=1e-7,
        )

        return completion.choices[0].message.content.replace("\n", "")


class FluentsExtractor:
    def __init__(self, **kwargs):
        self.fluents = kwargs["fluents"]
        self.objects = kwargs["objects"]
        self.init = INIT_STATE
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = "gpt-4o"
        self.completion = None

    def __call__(self, sentence):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistent which is very capable in translating natural language to PDDL fluent. \
                    You know that all the possible fluents that can occur in this domain are {self.fluents}, where they appear \
                    in the form of (fluent x - type_of_x y type_of_y), or (fluent x - type_of_x) or (fluent). \
                    All and only the variables admitted are the followings: {self.objects}. \
                    You know that the initial state is the following: {self.init}. \
                    If something has not been explicitely addressed in the sentence, you can assume that it is not changed from the initial state. \
                    Remember that to reach a certain location, you have to have cleared the previous ones. \
                    Your goal is to provide me a set of fluents used ONLY the fluents and objects that I provided you, \
                    given a sentence in a natural language. When answering, do not provide any explanation, just the set of fluents. \
                    When answering, ignore the type of the objects in the fluents.\
                    Also, provide them in the following structure '(fluent1),(fluent2),(fluent3),...' and so on.",
                },
                {
                    "role": "user",
                    "content": "The sentence to transform is: 'I have just finished to check the grape \
                        at location l2. After handling this grape, I am going to move to location l3 and check the grape there.'",
                },
                {
                    "role": "system",
                    "content": "(adj l0 l1),(adj l1 l2),(adj l2 l3),(robot-at rob l2),(unchecked l3),(in b rob),(full rob),(robot-at support0 l0),(support support0),(grape-at g3 l3),(free rob)",
                },
                {
                    "role": "user",
                    "content": "The sentence to transform is: 'I am in location l2 and I have called the support robot to\
                        empty my box because it is full in order to collect the ripe grape.",
                },
                {
                    "role": "system",
                    "content": "(adj l0 l1),(adj l1 l2),(adj l2 l3),(robot-at rob l2),(full b),(ripe g2),(robot-at support0 l2),(support support0),(grape-at g2 l2),(grape-at g3 l3),(free rob)",
                },
                {
                    "role": "user",
                    "content": f"The sentence to transform is: '{sentence}' "
                }
            ],
            stream=False,
            temperature=1e-7,
        )

        return completion.choices[0].message.content.replace("\n", "").replace("\t", "")
