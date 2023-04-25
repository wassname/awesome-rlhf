
This is a list of resources for reinforcement learning from human feedback (RLHF) and other methods to instruct large language models.



## Data

Data can generally be divided along two axis:

- high quality 🗹 or Lower quality ☐
- natural 🧑 or unnatural 🤖

Depending on your training objectives you will want lots of low quality instruction data, or a small amount of high quality data. Which should you use? Lets see what Anthropic have to say in [Askell et al](https://arxiv.org/abs/2112.00861): 

> **How can we improve the sample efficiency of preference modeling?** We find that we can significantly improve sample efficiency using a ‘preference model pre-training’ (PMP) stage of training, where we first pre-train on large public datasets that encode human preference information, such as Stack Exchange, Reddit, and Wikipedia edits, before finetuning on smaller datasets encoding more specific human preferences.

### Natural 🧑 & High quality 🗹

- OASST1- [OpenAssistant Conversations Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) 160k rows, 2023-04-12
- SHP - [Stanford human preferences](https://huggingface.co/datasets/stanfordnlp/SHP) - a dataset of instructions inferred from high quality sbureddits. 300k rows. 2023-02-23  [tweet](https://twitter.com/ethayarajh/status/1628442009500524544/photo/1)
- [HH-RLHF - Antropic RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) 91k rows
- [allenai/natural-instructions](https://github.com/allenai/natural-instructions) 64k rows
- [hendrycks/ethics](https://github.com/hendrycks/ethics) 130k rows

### Natural 🧑 & Lower quality ☐


-  ELI5: a reddit based dataset of questions and answers. The [SHP](https://huggingface.co/datasets/stanfordnlp/SHP) dataset improved on it's processing by comparing score *and* time
- https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences 10M instruction stack exchange, was used in anthropic paper  [paper](https://arxiv.org/abs/2112.00861)]: 

### Unnatural 🤖 & High quality  🗹

- [`alpaca_data_cleaned.json`](https://github.com/gururise/AlpacaDataCleaned) GPT4 instruction data, with heavy curation
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



## Training

### Libraries

- https://github.com/lucidrains/PaLM-rlhf-pytorch - Implementation of RLHF on top of the PaLM 
- https://github.com/CarperAI/trlx - A repo for distributed training of language models with Reinforcement Learning via Human Feedback (RLHF) 

- https://huggingface.co/docs/trl/index transformer reinforcement learning
- https://github.com/allenai/RL4LMs
- https://github.com/voidful/TextRL

### tutorials

- https://huggingface.co/blog/stackllama - StackLLaMA: A hands-on guide to train LLaMA with RLHF 
- https://huggingface.co/blog/rlhf


### Papers/Methods

- [RLHF](https://arxiv.org/pdf/2009.01325.pdf)
- Chain of Hindsight https://arxiv.org/abs/2302.02676 the model it trained to rank it's own output, so it's kind of like diffusion, letting the model operate iterativly. 
- SFT - Supervised Fine Tuning this is normal fine tuning
- [Pretraining Language Models with Human Preferences](https://arxiv.org/abs/2302.08582) [tweet](https://twitter.com/tomekkorbak/status/1628088313252052993?lang=en) You can (and should) do RL from human feedback during pretraining itself! In our new paper, we show how training w/ human preferences early on greatly reduces undesirable LM behaviors
- HIR: [Hindsight Instruction Relabeling](https://twitter.com/tianjun_zhang/status/1628180891368570881) 💩 offline RL reinvented with extra steps
 FARL: SL
  Algorithm Distillation: classical control problems. Offline RL
- Hindsight Instruction Relabeling (HIR), https://arxiv.org/abs/2302.05206 " outperforms the baseline algorithms and is comparable to or even surpasses supervised finetuning. "
  
  
 
## Evaluation

There are multiple ways to formally evaluate LLM capabilities. Right now project generally use one of these 3 libraries. Personally I prefer Eleuther's work, but opinions and github stars are divided.

- python api:
	- [huggingface/evaluate](https://github.com/huggingface/evaluate) this is not specific to LLM's or RLHF, but [some](https://github.com/nomic-ai/gpt4all/blob/main/eval_self_instruct.py#L43) [projects](https://github.com/gururise/AlpacaDataCleaned/blob/791174f63e/eval/README.md) find it and easy to use starting point. 
- cli api:
	- [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - has lots of datasets like GLUE and ETHICS already included, works with huggingface
	- [openai/evals: Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.](https://github.com/openai/evals) - has lots of rare eval sets like sarcasm, works with langchain
	- [stanford-crfm/helm: Holistic Evaluation of Language Models (HELM), a framework to increase the transparency of language models (https://arxiv.org/abs/2211.09110).](https://github.com/stanford-crfm/helm), works with huggingface


## Similar lists

- very comprehensive list https://github.com/yaodongC/awesome-instruction-dataset :star:
- divides the data in a similar way https://github.com/raunak-agarwal/instruction-datasets
- has tables https://github.com/zhilizju/Awesome-instruction-tuning
- papers https://github.com/SinclairCoder/Instruction-Tuning-Papers

