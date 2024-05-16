---
author: ["Navid Madani"]
title: "Enabling Reasoning Capabilities in Large Language Models - Part 2"
date: "2024-03-15"
tags: ["Reasoning"]
ShowToc: true
draft: false
TocOpen: true
---


# Prior Limitations and Design Choices

In [part 1](https://navidmdn.github.io/homepage/blog/reasoning-verifier-guided-search-emnlp22/) of this series, we talked
about the verifier-guided search method which was introduced in [this paper](https://arxiv.org/abs/2205.12443). We
discussed different components of their proposed method (prover, verifier and the search algorithm) and how they work and
in the end we talked about possible design flaws and rooms for improvement. In this part, I want to design my own method
for reasoning and try to overcome the limitations of the previous method.



Recent advancements in natural language processing (NLP) have driven the development of increasingly powerful language models.
These models are capable of performing a wide range of tasks with high accuracy, assuming they are provided with well-defined objectives and are trained using comprehensive datasets. This assumption forms the foundation of the design choices in this work.
The NLProofs method, discussed in Part 1, exhibits significant shortcomings, particularly in defining clear objectives and
maintaining simplicity. These issues are addressed in the current work. One of the critical challenges in NLProofs is
the design of the stepwise deductor, which is inherently complex. This complexity can confuse the model, making the 
learning task more difficult. In the stepwise deduction process of NLProofs, the model is tasked with both selecting
relevant premises and performing the deduction step. This dual requirement can overwhelm the model,
leading to inefficiencies and inaccuracies. To address this, I propose a new model design that separates the selection 
and deduction steps. By decoupling these processes, the model becomes more modular, with each component having a 
clear and distinct objective. This separation not only simplifies the learning task for the model but also enhances
explainability. By having dedicated modules for selection and deduction, we can better understand and interpret 
the model's behavior and reasoning process. This modular approach aims to improve the overall performance and
reliability of the system, making it more robust and easier to analyze.

## Selection Module

The selection module is responsible for learning the task of selecting feasible forward reasoning premises. When we want
to solve the problem ourselves using forward reasoning, we usually have a set of premises, and we select the most relevant
ones to combine while have an eye on the final goal (which kind of mimics the backward reasoning capabilities).
With this approach in mind we can design a selection module that takes current state of the reasoning process and the
final goal and selects the most relevant premises to combine.
Note that the selection module only selects the premises and does not combine them. The combination of the premises is
done in a separate module. We try to isolate the tasks as much as possible to make the reasoning process more modular. This
way we'll be able to evaluate and debug each of the modules more effectively.

### selection task details

Now, the question is what is a good neural architecture for this task? Let's first define the input and output of the
selection module. The input to the selection module is a set of premises and the final goal. The output is a set of selected
premise identifiers as shown in the figure below.

![selector-data](/homepage/images/selector_inp_out.png)

While working on this project, I also noticed another work on building modular reasoning systems called [MetGEN](https://aclanthology.org/2022.findings-naacl.145/)
In their work, they also have a selection module that selects the most relevant premises to combine. They use a transformer based
encoder only model to gather embeddings out of each premise and then use FFNN to combine each two embeddings and score possible
combinations as shown in the figure below. One straightforward problem with this approach is that it is not scalable in case we may have to combine multiple
premises at one step. In that case, unlike how **a person** would do, we have to iteratively combine each two at a time to build up
the that one selection step. 

![Metgen](/homepage/images/metgen-selector.png)

Now, getting back to the hypothesis that LLMs are capable and we only need to guide them correctly, let's try to design a 
system that selects as many premises as possible at once while this is the only task it does. One possible approach to avoid the
complexity of searching through different combinations but still be able to select multiple premises at once is to use a 
generative model. We can train a generative model to generate a set of selected premises given the input premises and the final goal.
So for this step we chose a sequence to sequence transformer model (more specifically Flan-T5) model to tackle this task.
Since Flan models are trained to follow instructions and generate text, they can be a good fit for this task.

### Do we have a perfect objective?

Now, let's think deeper about the objective of the selection module. The selection module should select the most relevant
premises to combine, but there are multiple problems with this objective. First, there may be multiple correct solutions
as we might be able to start building up entailment tree (bottom-up) starting from different leaf nodes. Second, the objective
itself is not well defined. Let's assume the label for the selection task is **sent1 & sent4 & int2**. If you think about 
the task, the loss function should be invariant to the order of the selected premises. So, the model should be able to select
whichever premise first and then the other ones. But, the model may not be able to learn this invariance if we only use the
negative log likelihood loss. So, we need to come up with a better loss function that is invariant to the order of the selected
premises.

Having these issues in mind, to design a first version of our approach we only mitigate the issue partially, by 
randomizing the ids of the premises and shuffling them before feeding them to the model. This way, the model should be able
to understand that the order of the selected premises does not matter and the content of the premises is what matters.
It is not a perfect solution but it is a good start to see if the model can learn the task. 

### validation of the selection module

To validate the selection module and make sure the initial model is able to learn the task and most importantly the fact
that order of the selection doesn't matter, while training, at evaluation time, we generate multiple continuations using 
deterministic beam search algorithm to take top-k continuations and evaluate if one of them is the correct one. We call this
metric **top-k accuracy**. This metric is a good indicator of how well the model is able to learn the task and how well it
is able to ignore the order of the selected premises. This way, we select the best model based on this metric and avoid
overfitting to the training data.

![selector-flan](/homepage/images/selector-flan.png)

## Deduction Module

The deduction module is responsible for combining the selected premises together to form a new premise. I tried to train
this module in a way that is independent of selection and the hypothesis that we want to prove. In the most general case,
it takes a set of premises and tries to deduct a new premise from them.

## Proof Search