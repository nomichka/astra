import openai
import os
import time
from dotenv import load_dotenv

# Load and set up OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

TRAIN_FILE = "test_train_finetune.jsonl"
TEST_FILE = "test_finetune.jsonl"


def create_finetune_job(
    client, model, train_id, suffix, test_id=None, hyperparameters={"n_epochs": 1}
):
    return client.fine_tuning.jobs.create(
        training_file=train_id,
        model=model,
        validation_file=test_id,
        suffix=suffix,
        hyperparameters=hyperparameters,
    )


# Function to check the status of the fine-tuning job
def is_job_complete(client, job_id):
    job_details = client.fine_tuning.jobs.retrieve(job_id)
    return job_details.status == "succeeded"


# Function to list models with the specified suffix
def list_models_with_suffix(client, suffix):
    models = client.fine_tuning.jobs.list()
    filtered_models = [
        model for model in models["data"] if model["id"].endswith(suffix)
    ]
    return filtered_models


def main():
    client = openai.OpenAI(api_key=openai.api_key)

    train_id = client.files.create(file=open(TRAIN_FILE, "rb"), purpose="fine-tune")
    test_id = client.files.create(file=open(TEST_FILE, "rb"), purpose="fine-tune")
    suffix_start = "test-step-1"

    job = create_finetune_job(
        client,
        train_id=train_id.id,
        model="gpt-3.5-turbo",
        suffix=suffix_start,
    )
    # n_epochs=1, n_examples=1, n_batch=1, n_validation=1, stop="Validation loss: "

    job_id = job.id

    # Polling the fine-tuning job status
    job_details = client.fine_tuning.jobs.retrieve(job_id)
    while job_details.status != "succeeded":
        print("Waiting for fine-tuning job to complete...")
        time.sleep(15)  # Wait for 60 seconds before checking again
        job_details = client.fine_tuning.jobs.retrieve(job_id)

    print("Fine-tuning job completed.")

    # Get model id
    model = job_details.fine_tuned_model

    create_finetune_job(client, train_id=test_id.id, model=model, suffix="second-stage")


if __name__ == "__main__":
    main()
