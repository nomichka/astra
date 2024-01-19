# %%
import openai
import os
import random
import time
import math
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# %%

TRAIN_FILE = "test_train_finetune.jsonl"
TEST_FILE = "test_finetune.jsonl"


class finetune:
    def __init__(self):
        # Load and set up OpenAI API key
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        openai.api_key = OPENAI_API_KEY
        self.model = model
        self.client = openai.OpenAI(api_key=openai.api_key)

    # %%

    def create_finetune_job(
        self, model, train_id, suffix="", test_id=None, hyperparameters={"n_epochs": 3}
    ):
        return self.client.fine_tuning.jobs.create(
            training_file=train_id,
            model=model,
            validation_file=test_id,
            suffix=suffix,
            hyperparameters=hyperparameters,
        )


    # Function to check the status of the fine-tuning job
    def is_job_complete(self, job_id):
        job_details = self.client.fine_tuning.jobs.retrieve(job_id)
        return job_details.status == "succeeded"


    # Function to list models with the specified suffix
    def list_models_with_suffix(self, suffix):
        models = self.client.fine_tuning.jobs.list()
        filtered_models = [
            model for model in models["data"] if model["id"].endswith(suffix)
        ]
        return filtered_models
    

# get ChatGPT-4 guess
def get_completion(self, history, model="gpt-3.5-turbo"):
    chat_completion = self.client.chat.completions.create(
        messages=history, model=model, temperature=0
    )
    response = chat_completion.choices[0].message.content
    print(f"(Model completion with {model}) {response}")
    return response


def get_logprobs(history, model="gpt-3.5-turbo", num_logprobs=4):
    chat_completion = self.client.chat.completions.create(
        messages=history,
        model=model,
        logprobs=True,
        top_logprobs=num_logprobs,
        temperature=0,
    )
    response = chat_completion.choices[0].logprobs.content[0].top_logprobs
    tokens_and_probs = []
    for choice in response:
        token_one = choice.token
        prob_one = math.exp(choice.logprob)
        tokens_and_probs.append((token_one, prob_one))

    print(f"(Model logprobs with {model}) {tokens_and_probs}")
    return tokens_and_probs
