---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: A step-by-step guide for choosing among training techniques. TODO: polish this description
---

# Choosing the Right Training Technique

TODO: polish and extend(?) this intro
TODO: polish title. include something about LLMs?
This is a guide for choosing the best technique to train a model on your data. We'll list some techniques here

## Table of Contents

TODO: build this


## Prompt Engineering

### Few-Shot Learning (FSL)

### CoT

## Finetuning

### Cutting Few-Shot Costs and Latency

TODO: show GPT-4 cost curves here with FSL, inferencing volume, Twitter dataset. Mention that accuracy consistent but price is not
TODO: latency study and/or experiments?

### Altering Model Behavior (TODO: come up with a better name)

* Learning a new skill
* "Show not tell"
  * Too many edge cases

Nuance dataset, chess dataset, describe Bing dataset

### Model distillation

https://arxiv.org/abs/2305.02301 + any repro

### Reduce hallucination

https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/ft_retrieval_augmented_generation_qdrant.ipynb + any repro

### C

## RAG

### When fact retrieval is helpful

### When selecting similar examples is helpful

Cite: hallucination study above

### Stacking Techniques

## MSR Ag study

## Todo: Cite hallucination study above
