---
author: ["Navid Madani"]
title: "Efficient Fine-tuning of Llama3 with QLoRA"
date: "2024-06-18"
tags: ["LLMs", "Fine-tuning", "peft"]
ShowToc: true
draft: false
TocOpen: true
---

You can find all the experiments and code for this part in the following github repo.
<div align="center">
	<a href="https://github.com/navidmdn/llm_recipes/blob/main/llama3_qlora_sft.py">
	    <img src="/homepage/images/github-mark.png" width="40" height="40" alt="Github">
	</a>
</div>

# Introduction

At the time of writing this blog post, I was struggling to find any comprehensive resources for
fine-tuning Llama3 using QLoRA, especially those that included the key features I needed.
This tutorial aims to bridge that gap by providing a detailed walkthrough, explaining the
nuances and challenges I encountered, and offering a complete script to help you develop your own use case.

This guid does not go into the details of the concepts behind LLMs, QLoRA, or fine-tuning in general. 
It assumes you have a basic understanding of these topics and focuses on the practical aspects of fine-tuning Llama3 with QLoRA.

We will use the `peft`, `trl` and `transformers` libraries to build our fine-tuning pipeline. The full script can
be downloaded from [this repo](https://github.com/navidmdn/llm_recipes/blob/main/llama3_qlora_sft.py). I'll go over
important parts of it and explain a few important features that I needed to have in my pipeline.

A set of important features that I needed to have in my pipeline are:
- lightweight local testing of the model
- using `trl`'s `SFTTrainer` to avoid implementing every little detail that we might miss
- how to tokenize and preprocess your custom instruction dataset for **instruction completion**
  - most of the available tutorials only consider fine-tuning over the whole corpus and do not consider the instruction completion task without calculating loss over instruction as well 
- supporting multiple validation sets and monitoring training progress separately over them

# preparing your dataset

The first step is to prepare your dataset. You can use any dataset you like, and it is one important part of the process.
The thing is, that the design and specific of all these libraries keep changing over time and at the time of writing this
post, there was not an official best practice to use your custom dataset for my use case. Most importantly, I was interested
in only instruction completion and not calculating loss over the instruction as well. There were a few tutorials that I found
on this topic, but they were not comprehensive enough to be used with llama3 instruction tuned version and they lacked some
nuances that are very important when you're training a instruction tuned model.


let's assume we have a train and validation dataset in json line format with the following structure:

```json
{
  "query": "this is a sample instruction",
  "response": "this is a sample text"
}
```

the training file will be loaded from `script_args.train_file` and the validation files will be loaded from `script_args.dev_dir` and
each validation file name should be ended in `val.json`. This way we can load multiple validation files and monitor the training progress
on each of them separately. **It is very important when we want to measure how good the model has learned different skills 
during the training process and a requirement for me**.
  
  
```python
train_dataset = load_dataset(
    "json",
    data_files=script_args.train_file,
    split="train",
    cache_dir=script_args.cache_dir,
)

dev_datasets = {}
dev_files = glob.glob(os.path.join(script_args.dev_dir, "*val.json"))
print(f"list of validation files: {dev_files}")

if len(dev_files) > 0:
    for dev_file_path in dev_files:
        print('loading ', dev_file_path)
        ds = load_dataset(
            "json",
            data_files=dev_file_path,
            split="train",
            cache_dir=script_args.cache_dir,
        )
        dev_split_name = dev_file_path.split("/")[-1].split(".")[0]
        dev_datasets[dev_split_name] = ds
else:
    raise NotImplementedError("No dev files provided")
```

Afterwards, according to your data scheme, write a preprocessing function that tokenizes your data:

```python
def preprocess_dataset(example):
    """
    Preprocess the dataset to convert it to chat format
    :param example:
    :return:
    """
    full_chat_formatted = [
        {"role": "system", "content": "You are a helpful and friendly assistant."},
        {"role": "user", "content": example['query']},
        {"role": "assistant", "content": example['response']}
    ]

    user_prompt_chat_formatted = [
        {"role": "system", "content": "You are a helpful and friendly assistant."},
        {"role": "user", "content": example['query']},
    ]

    full_chat_input_ids = tokenizer.apply_chat_template(
            full_chat_formatted,
            add_generation_prompt=False,
            tokenize=True
    )

    input_chat_input_ids = tokenizer.apply_chat_template(
            user_prompt_chat_formatted,
            add_generation_prompt=True,
            tokenize=True
    )

    labels = np.array(full_chat_input_ids)
    labels[:len(input_chat_input_ids)] = -100

    return {
        'input_ids': np.array(full_chat_input_ids),
        'labels': labels,
    }
```

The `tokenizer.apply_chat_template` function will take care of the specific formatting that the underelying model (llama3 here)
requires. Along with this, we define a data collator to pad the input sequences to the longest sequence length in the batch:

```python
class Llama3ChatCompletionDataCollator(DataCollatorMixin):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, return_tensors='pt'):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        raw_input_ids: List[np.ndarray] = [e['input_ids'] for e in examples]
        raw_labels: List[np.ndarray] = [e['labels'] for e in examples]
        max_input_len = max([len(p) for p in raw_input_ids])

        labels = []
        input_ids = []
        attention_masks = []
        for full_input_ids, label_ids in zip(raw_input_ids, raw_labels):
            pad_len = max_input_len - len(full_input_ids)
            pad_list = [self.tokenizer.pad_token_id]*pad_len
            ignore_loss_list = [-100]*pad_len

            label_ids = np.concatenate([label_ids, ignore_loss_list])
            attention_mask = [1] * len(full_input_ids) + [0] * pad_len
            full_input_ids = np.concatenate([full_input_ids, pad_list])

            labels.append(label_ids)
            input_ids.append(full_input_ids)
            attention_masks.append(attention_mask)

        return {
            'input_ids': torch.LongTensor(np.array(input_ids)),
            'labels': torch.LongTensor(np.array(labels)),
            'attention_mask': torch.LongTensor(np.array(attention_masks))
        }
```

One very important thing here, is that llama3 does not have a `pad_token_id` in its tokenizer. So, we need to add it manually
so we also add the following lines to set the `pad_token_id` to an extra token available in it's tokenizer:

```python
tokenizer.pad_token = "<|reserved_special_token_4|>"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
```
Many of the tutorials that I found used the `eos_token` instead, but for some models that actually use the `eos_token`,
this will cause some problems to separate the actual padding token from the `eos_token` later (or at least it was puzzling for me).

# A custom evaluation function

Another important feature that I needed to have in my pipeline was to have a custom evaluation function that can be used
to generate responses to evaluation samples and calculate the metrics over them just like what we do with the `Seq2SeqTrainer`
in transformers library. To do so, we need to pass the `include_inputs_for_metrics` to your training arguments to get the 
inputs to the model while the trainer calls `compute_metrics` function. This way we can generate responses to the inputs
ourselves. That is how we implement the `compute_metrics` function in our script. 

An important thing to note here is that while we are generating responses to the inputs, we need to make sure that the inputs
are padded on the left side, so we make sure to switch the padding side to the left as follow:

```python
tokenizer.padding_side = "left"
```

The rest of the script is straight forward and uses general tutorials for setting up the training pipeline which uses
`transformers`, `trl` and `peft` libraries. You can find the full script in the github repo above.


