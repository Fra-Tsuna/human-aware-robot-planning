<div align="center">
<h1 style="font-size: 20px">Human-Robot Awareness in<br> Collaborative Scenarios &#x1F347; <br> 
Combining LLMs and Symbolic Reasoning</h1>
<div>



</div>
</div>

## Extra Material
For a demonstration of our system in use in the simulation scenario of a real-life agricultural use case (anonimized 
to respect the submission guidelines), please check the video in the `assets` folder.

## Prerequisites
Clone this repo and its submodule
```
git clone --recurse-submodules https://github.com/Fra-Tsuna/human-aware-robot-planning.git
```
Install the necessary Python dependencies:
```
conda create -n <YOURENV> python=3.9 dill matplotlib plotly scipy scikit-learn pandas tenacity
conda activate <YOURENV>
pip install openai tiktoken
pip install openai --upgrade
pip install nltk seaborn pyyaml
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
conda install pytorch torchdata -c pytorch  # CPU
pip install tensorboard transformers datasets evaluate torchtext
conda install -c conda-forge spot
pip install sentencepiece
pip install chardet
```
Install Mona
```
sudo apt install mona
```
Install LTLf2DFA
```
pip install ltlf2dfa
```
Install FOND4LTLf
```
git clone --branch v0.0.3 https://github.com/whitemech/FOND4LTLf.git
cd FOND4LTLf
pip install .
```
Build PRP
```
cd planner-for-relevant-policies/src/
./build_all
```
Setup your OpenAI API key
```
conda-env config vars set OPENAI_API_KEY=<YOUR API KEY>
```
## Process Extraction
All the scripts to run the Process Extraction module are in the `process_extraction/` directory

### Model checkpoints
- Download the pretrained symbolic translation model from https://drive.google.com/drive/folders/1rZl8tblyVj-pZZW4OgbO1NJwMIT2fwx9
- Unzip and put the downloaded files into the directory `process_extraction/Lang2LTL/t5-base/composed_model_3000000/checkpoint-best/`

### Usage
To translate a single sentence in an LTL formula, run
```
cd process_extraction
python3 sentence_2_formula.py
```
The script accepts some parameters. You can read the list of parameters by running `python sentence_2_formula.py --helpshort`.
```
sentence_2_formula.py:
  --LANDMARKS_FILE: json file with landmarks
    (default: 'landmarks.json')
  --LOG_FILE: csv file where to write the results
    (default: 'results/result.csv')
  --RER_PROMPT_FILE: txt file containing the rer prompt
    (default: 'rer_prompt_augmented_2.txt')
  --SENTENCE: Sentence to translate in LTL
    (default: 'go to line 1')
```
To reproduce the process extraction evaluation results we provide in the paper, run:
```
cd process_extraction
python3 run_experiments.py
```
Here's the list of flags accepted by the script
```
run_experiments.py:
  --DATASET: csv file containing sentences and target LTL formulas
    (default: 'dataset/DS_all_symbols.csv')
  --LANDMARKS_FILE: json file with landmarks
    (default: 'landmarks.json')
  --LOG_FILE: csv file where to write the results
    (default: 'results/result.csv')
  --RER_PROMPT_FILE: txt file containing the rer prompt
    (default: 'rer_prompt_augmented_2.txt')
```

## Planning and Process Awareness
All the scripts to run the Planning and Process Awareness are in the `planning/` directory

### Usage
To integrate the LTLf goal inside the planning domain, run:
```
fond4ltlfpltlf -d config/PDDL/domain.pddl -p config/PDDL/problem.pddl -g "<formula>" --out-domain config/PDDL/ltlf_domain.pddl --out-problem config/PDDL/ltlf_problem.pddl
```
For example, a possible LTLf formula could be "F(cleared_l3)", to ensure that location l3 will be eventually cleared.

To obtain the policy, first it is necessary to remove "or" preconditions from the newly obtained domain by splitting an action with such preconditions into 2 or more actions with the "or" resolved among them.
E.g.:
```
(:action trans-1
  :parameters (?l-00 - location)
  :precondition (and (or (and (q1 ?l-00) (cleared ?l-00)) (q2 ?l-00)) (not (turnDomain)))
  :effect (and (q2 ?l-00) (not (q1 ?l-00)) (turnDomain))
)
```
should become:
```
(:action trans-10
  :parameters (?l-00 - location)
  :precondition (and (q1 ?l-00) (cleared ?l-00) (not (turnDomain)))
  :effect (and (q2 ?l-00) (not (q1 ?l-00)) (turnDomain))
)
(:action trans-11
  :parameters (?l-00 - location)
  :precondition (and (q2 ?l-00) (not (turnDomain)))
  :effect (and (q2 ?l-00) (not (q1 ?l-00)) (turnDomain))
)
```
To obtain the policy, run:
```
./scripts/obtain_policy.sh
```
To translate the policy into a human-readable format, run:
```
./scripts/make_policy_readable.sh
```
Lastly, we want to prepare the files to give in input to GPT4 to reduce allucinations. Run:
```
python3 planning/prepare_gpt_inputs.py
```

Since readers cannot test the system on real vineyards, we provided some scripts and tests that can executed to mimic the behaviour of the robot in the field (sas_plan_adapted).

To mimic the interaction of the human asking questions about the robot's actions, run
```
python3 planning/plan_simulation.py
```
The user has a window of 10 seconds after every robot action to make any kind of questions to the system and observe its response.

To reproduce the experiments of the paper on the awareness, run
```
python3 planning/awareness_experiments.py --n_exp 30
```
