
This is a list of resources for reinforcement learning from human feedback and other methods to instruct large language models.

## Evaluation

There are multiple ways to formally evaluate LLM capabilities. Right now project generally use one of these 3 libraries. Personally I prefer Eleuther's work, but opinions and github stars are divided.

- [EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of autoregressive language models.](https://github.com/EleutherAI/lm-evaluation-harness)
- [openai/evals: Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.](https://github.com/openai/evals)
- [stanford-crfm/helm: Holistic Evaluation of Language Models (HELM), a framework to increase the transparency of language models (https://arxiv.org/abs/2211.09110).](https://github.com/stanford-crfm/helm)

## Training



## Data

Data can generally be divided along two axis:

- high quality 🗹 or Lower quality ☐
- natural 🧑 or unnatural 🤖

Depending on your training objectives you will want lots of low quality instruction data, or a small amount of high quality data. Which should you use? Lets see what Anthropic have to say [Askell et all Antrhopic](https://arxiv.org/abs/2112.00861)]: 

> **How can we improve the sample efficiency of preference modeling?** We find that we can significantly improve sample efficiency using a ‘preference model pre-training’ (PMP) stage of training, where we first pre-train on large public datasets that encode human preference information, such as Stack Exchange, Reddit, and Wikipedia edits, before finetuning on smaller datasets encoding more specific human preferences.

### Natural 🧑 & High quality 🗹

- oasst- [from open assistant]([https://huggingface.co/OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1)) 22k rows, 2023-04-12
- SHP - [Stanford human preferences](https://huggingface.co/datasets/stanfordnlp/SHP) - a dataset of instructions inferred from high quality sbureddits. 300k rows. 2023-02-23  [tweet](https://twitter.com/ethayarajh/status/1628442009500524544/photo/1)
- [HH-RLHF - Antropic RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) 91k rows
- https://github.com/allenai/natural-instructions 64k rows
- https://github.com/hendrycks/ethics 130k rows

### Natural 🧑 & Lower quality ☐


-  ELI5: a reddit based dataset of questions and answers. The [SHP](https://huggingface.co/datasets/stanfordnlp/SHP) dataset improved on it's processing by comparing score *and* time
- https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences 10M instruction stack exchange, was used in anthropic paper  [paper](https://arxiv.org/abs/2112.00861)]: 

### Unnatural 🤖 & High quality  🗹

- [`alpaca_data_cleaned.json`](https://github.com/tloen/alpaca-lora/pull/32) including removing as a large language model https://github.com/gururise/AlpacaDataCleaned
- https://github.com/teknium1/GPTeacher
- https://github.com/databrickslabs/dolly
- [OIG-small-chip2](https://laion.ai/blog/oig-dataset/) a subset of the OIG dataset

### Unnatural 🤖 & Lower quality ☐

- [unnatural-instructions](https://github.com/orhonovich/unnatural-instructions) used above and GPT3 to make 256k examples
- OIG - [Open Instruction Generalist Dataset](https://laion.ai/blog/oig-dataset/) a compilation of ~43M instructions. "The OIG dataset is almost purely a synthetic data set created using data augmentation.""
	- note there is a higher quality subset OIG-small-chip2

### Uncategorized

- b-mc2 (https://huggingface.co/datasets/b-mc2/wikihow_lists).


### Finding more data

A great way to find new instruction datasets is to
- [search huggingface's datasets](all hf data [1](https://huggingface.co/search/full-text?q=rlhf&type=dataset))
- Look at compilations like - [OIG](https://laion.ai/blog/oig-dataset/)
- [github instruction-turning tag](https://github.com/topics/instruction-tuning)

Similar lists

- https://github.com/yaodongC/awesome-instruction-dataset
- https://github.com/zhilizju/Awesome-instruction-tuning
- https://github.com/raunak-agarwal/instruction-datasets