"""
THIS FILE CONTAINS.

1. DATASETS INGESTION FUCNTIONS
2. PROMPT GENERATOR CLASSES
3. AZURE MODEL INFERENCE CONNECTOR HELPER CLASSES
4. FINAL DATASET PUBLISHING CLASSES

"""
from abc import ABC
from datasets import load_dataset
import json
import requests


"""
DATASETS INGESTION FUNCTIONS

"""


class InputDataset(ABC):
    """Input Dataset class."""

    def __init__(self):
        """Initialize the class."""
        super().__init__()
        (
            self.train_data_file_name,
            self.test_data_file_name,
            self.eval_data_file_name,
        ) = (None, None, None)


class QALocalInputDataset(InputDataset):
    """
    Loads the input dataset if its in local.

    The directory is left blank if your dataset is in the same directory as your notebook.
    The input dataset is divided as train, eval and test based on availaibility
    """

    def __init__(
        self,
        dir="",
        dataset_name="cqa",
        train_samples="512",
        test_samples="256",
        eval_samples=None,
    ):
        """Initialize the class."""
        super().__init__()
        self.dir = dir
        if train_samples is not None:
            self.train_data_file_name = (
                dir + dataset_name + "_train_" + str(train_samples) + ".jsonl"
            )
        if test_samples is not None:
            self.test_data_file_name = (
                dir + dataset_name + "_test_" + str(test_samples) + ".jsonl"
            )
        if eval_samples is not None:
            self.eval_data_file_name = (
                dir + dataset_name + "_eval_" + str(eval_samples) + ".jsonl"
            )

    def load_local_dataset(self, sample_size=10):
        """Load the local dataset."""
        train_data, val_data, test_data = [], [], []

        if self.train_data_file_name:
            with open(self.train_data_file_name, "r") as f:
                for line in f:
                    train_data.append(json.loads(line))

        if self.test_data_file_name:
            with open(self.test_data_file_name, "r") as f:
                for line in f:
                    test_data.append(json.loads(line))

        if self.eval_data_file_name:
            with open(self.eval_data_file_name, "r") as f:
                for line in f:
                    val_data.append(json.loads(line))
        return train_data[:sample_size], val_data[:sample_size], test_data[:sample_size]


class QAHuggingFaceInputDataset(InputDataset):
    """Loads the HuggingFace dataset."""

    def __init__(self):
        """Initialize the class."""
        super().__init__()

    def load_hf_dataset(
        self,
        dataset_name,
        train_sample_size=10,
        val_sample_size=10,
        test_sample_size=10,
        train_split_name="train",
        val_split_name="validation",
        test_split_name="test",
    ):
        """Load the HuggingFace dataset."""
        full_dataset = load_dataset(dataset_name)

        if val_split_name is not None:
            train_data = full_dataset[train_split_name].select(range(train_sample_size))
            val_data = full_dataset[val_split_name].select(range(val_sample_size))
            test_data = full_dataset[test_split_name].select(range(test_sample_size))
        else:
            train_val_data = full_dataset[train_split_name].select(
                range(train_sample_size + val_sample_size)
            )
            train_data = train_val_data.select(range(train_sample_size))
            val_data = train_val_data.select(
                range(train_sample_size, train_sample_size + val_sample_size)
            )
            test_data = full_dataset[test_split_name].select(range(test_sample_size))

        return train_data, val_data, test_data


class NLIHuggingFaceInputDataset(InputDataset):
    """Loads the HuggingFace dataset."""

    def __init__(self):
        """Initialize the class."""
        super().__init__()

    def load_hf_dataset(
        self,
        dataset_name,
        train_sample_size=10,
        val_sample_size=10,
        test_sample_size=10,
        train_split_name="train",
        val_split_name="validation",
        test_split_name="test",
    ):
        """Load the HuggingFace dataset."""
        full_dataset = load_dataset(dataset_name)

        if val_split_name is not None:
            train_data = full_dataset[train_split_name].select(range(train_sample_size))
            val_data = full_dataset[val_split_name].select(range(val_sample_size))
            test_data = full_dataset[test_split_name].select(range(test_sample_size))
        else:
            train_val_data = full_dataset[train_split_name].select(
                range(train_sample_size + val_sample_size)
            )
            train_data = train_val_data.select(range(train_sample_size))
            val_data = train_val_data.select(
                range(train_sample_size, train_sample_size + val_sample_size)
            )
            test_data = full_dataset[test_split_name].select(range(test_sample_size))

        return train_data, val_data, test_data


"""
PROMPT GENERATOR CLASSES
"""


class PromptGenerator(ABC):
    """Prompt Generator class."""

    def __init__(self):
        """Initialize the class."""
        super().__init__()

    def generate_prompt(self):
        """Generate the prompt."""
        pass


class QAPromptGenerator(PromptGenerator):
    """Prompt format each data for inference."""

    def __init__(self):
        """Initialize the class."""
        super().__init__()
        self.qa_system_prompt = (
            "You are a helpful assistant. Write out in a step by step manner "
            "your reasoning about the answer using no more than 80 words. "
            "Based on the reasoning, produce the final answer. "
            "Your response should be in JSON format without using any backticks. "
            "The JSON is a dictionary whose keys are 'reason' and 'answer_choice'."
        )

        self.qa_user_prompt_template = (
            "Answer the following multiple-choice question by selecting the correct option.\n\n"
            "Question: {question}\n"
            "Answer Choices:\n"
            "{answer_choices}"
        )

    def generate_prompt(self, qa_input):
        """Generate the prompt."""
        _, choices, _ = qa_input["question"], qa_input["choices"], qa_input["answerKey"]

        labels, choice_list = choices["label"], choices["text"]
        answer_choices = [
            "({}) {}".format(labels[i], choice_list[i]) for i in range(len(labels))
        ]
        answer_choices = "\n".join(answer_choices)

        self.qa_user_prompt = self.qa_user_prompt_template.format(
            question=qa_input["question"], answer_choices=answer_choices
        )

        final_prompt = {
            "messages": [
                {"role": "system", "content": self.qa_system_prompt},
                {"role": "user", "content": self.qa_user_prompt},
            ]
        }

        return final_prompt


class NLIPromptGenerator(PromptGenerator):
    """Prompt format each data for inference."""

    def __init__(self, enable_chain_of_thought=False):
        """Initialize the class."""
        super().__init__()
        self.nli_user_prompt_template = (
            "Given the following two texts, your task is to determine the logical "
            "relationship between them. The first text is the 'premise' and the second "
            "text is the 'hypothesis'. The relationship should be labeled as one of the "
            "following: 'entailment' if the premise entails the hypothesis, 'contradiction' "
            "if the premise contradicts the hypothesis, or 'neutral' if the premise neither "
            "entails nor contradicts the hypothesis.\n\n"
            "Premise: {premise}\n"
            "Hypothesis: {hypothesis}"
        )
        if enable_chain_of_thought:
            self.nli_system_prompt = (
                "You are a helpful assistant. Write out in a step by step manner "
                "your reasoning about the answer using no more than 80 words. "
                "Based on the reasoning, produce the final answer. "
                "Your response should be in JSON format without using any backticks. "
                "The JSON is a dictionary whose keys are 'reason' and 'answer_choice'."
            )
        else:
            self.nli_system_prompt = (
                "You are a helpful assistant. "
                "Your output should only be one of the three labels: 'entailment', 'contradiction', or 'neutral'."
            )

    def generate_prompt(self, nli_input):
        """Generate the prompt."""
        premise, hypothesis = nli_input["premise"], nli_input["hypothesis"]

        self.nli_user_prompt = self.nli_user_prompt_template.format(
            premise=premise, hypothesis=hypothesis
        )

        final_prompt = {
            "messages": [
                {"role": "system", "content": self.nli_system_prompt},
                {"role": "user", "content": self.nli_user_prompt},
            ]
        }

        return final_prompt


"""
CONNECTION TO AZURE MODEL INFERENCING ENDPOINT CODE
"""


class AzureInference(ABC):
    """Azure Inference class."""

    def __init__(self, **kwargs):
        """Initialize the class."""
        super().__init__()

        self.url = kwargs["url"]
        self.key = kwargs["key"]

    def _invoke_endpoint(self, data):
        """Invoke the endpoint."""
        print(f"inferencing: {self.url}")

        response = requests.post(
            self.url,
            headers={
                "Content-Type": "application/json",
                "Authorization": ("Bearer " + self.key),
            },
            data=json.dumps(data),
        )

        return response

    def invoke_inference(self, prompt):
        """Invoke the inference."""
        response = self._invoke_endpoint(prompt)
        try:
            response_dict = json.loads(response.text)
            label = response_dict["choices"][0]["message"]["content"].strip().upper()
        except Exception as e:
            print(e)
            label = "error"
        return label


"""

SYNTHETIC DATASET BUILDER CLASSES

"""


class SyntheticDatasetBuilder(ABC):
    """Synthetic Dataset Builder class."""

    def __init__(self):
        """Initialize the class."""
        super().__init__()

    def _write_to_file(self, data, fname):
        """Write to file."""
        with open(fname, "w") as f:
            for sample in data:
                f.write(json.dumps(sample) + "\n")

    def _is_json(self, json_str):
        """Check if the string is a valid JSON."""
        try:
            json.loads(json_str)
        except ValueError:
            return False
        return True


class QASyntheticDatasetBuilder(SyntheticDatasetBuilder):
    """Builds dataset with Predicted labels by LLM."""

    def __init__(self, qa_prompt_builder, inference_pointer=None):
        """Initialize the class."""
        super().__init__()
        self.valid_answer_choices = ["A", "B", "C", "D", "E"]
        self.inference_pointer = inference_pointer
        self.qa_prompt_builder = qa_prompt_builder

    def build_dataset(
        self, dataset, file_name=None, write_labels_to_separate_file=False
    ):
        """Build the dataset."""
        if self.inference_pointer is not None or self.prompt_builder is not None:
            final_dataset = []
            no_label_dataset = []
            for row in dataset:
                prompt = self.qa_prompt_builder.generate_prompt(row)
                label = self.inference_pointer.invoke_inference(prompt)
                if self._is_json(label):
                    label = json.loads(label.strip()).get("ANSWER_CHOICE")
                if label not in self.valid_answer_choices:
                    continue
                new_content = {"role": "assistant", "content": label}
                no_label_dataset.append(prompt.copy())
                prompt["messages"].append(new_content)
                if write_labels_to_separate_file:
                    final_dataset.append(new_content)
                else:
                    final_dataset.append(prompt)
            if file_name is not None:
                if write_labels_to_separate_file:
                    self._write_to_file(
                        data=final_dataset, fname=file_name + "_label_" + ".jsonl"
                    )
                    self._write_to_file(
                        data=no_label_dataset, fname=file_name + ".jsonl"
                    )
                else:
                    self._write_to_file(data=final_dataset, fname=file_name + ".jsonl")

                print("Write to file complete")
            else:
                print("Please specify a valid endpoint first")


class NLISyntheticDatasetBuilder(SyntheticDatasetBuilder):
    """Builds dataset with Predicted labels by LLM."""

    def __init__(self, nli_prompt_builder, inference_pointer=None):
        """Initialize the class."""
        super().__init__()
        self.valid_labels = ["entailment", "contradiction", "neutral"]
        self.inference_pointer = inference_pointer
        self.nli_prompt_builder = nli_prompt_builder

    def build_dataset(
        self, dataset, file_name=None, write_labels_to_separate_file=False
    ):
        """Build the dataset."""
        if self.inference_pointer is not None or self.nli_prompt_builder is not None:
            final_dataset = []
            no_label_dataset = []
            for row in dataset:
                llm_body, llm_output = self._get_output_from_model(row)
                llm_body_copy = llm_body.copy()
                llm_body["messages"].append(llm_output)
                if write_labels_to_separate_file:
                    final_dataset.append(llm_body_copy)
                    no_label_dataset.append(llm_output)
                else:
                    final_dataset.append(llm_body)
            if file_name is not None:
                if write_labels_to_separate_file:
                    self._write_to_file(
                        data=no_label_dataset, fname=file_name + "_label_" + ".jsonl"
                    )
                    self._write_to_file(data=final_dataset, fname=file_name + ".jsonl")
                else:
                    self._write_to_file(data=final_dataset, fname=file_name + ".jsonl")
                print("Write to file complete")
            else:
                print("Please specify a valid endpoint first")

    def _get_output_from_model(self, data):
        """Get the output from the model."""
        prompt = self.nli_prompt_builder.generate_prompt(data)
        # Invoke the LLama endpoint
        label = self.inference_pointer.invoke_inference(prompt).lower()
        # Note that the following code block should be commented out if you are disabling CoT
        # ---- Start ----
        if self._is_json(label):
            label = json.loads(label.strip()).get("answer_choice")
        # --- End ---
        if label not in self.valid_labels:
            print("Invalid label generated by model")
            return
        new_content = {"role": "assistant", "content": label}

        return prompt, new_content
