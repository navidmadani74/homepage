---
author: ["Navid Madani"]
title: "Enabling Reasoning Capabilities in Large Language Models - Part 2"
date: "2024-03-15"
tags: ["Reasoning"]
ShowToc: true
draft: false
TocOpen: true
---

You can find all the experiments and code for this part in the following github repo.
<div align="center">
	<a href="https://github.com/navidmdn/llm_reasoning/tree/main/select_and_deduct">
	    <img src="/homepage/images/github-mark.png" width="40" height="40" alt="Github">
	</a>
</div>


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

The deduction module is more straightforward than the selection module since the only task it needs to do is logical 
deduction. The input to the deduction module is the selected premises. The output is the next step in the reasoning process.
There are some important considerations that I want to mention here. First, the deduction module should be able to handle
the cases were the selected premises are not enough to make a deduction. In that case, the model should be able to output 
a valid output and not hallucinate. One trick that I could think of to mitigate this issue is to augment the dataset with
some examples where the selected premises are not enough to make a deduction with the output indicating the logical AND of 
the selected premises. For example, if the selected premises are "the dog is red" and "the fox is wild" we augment the training
dataset with an example where the output is "the dog is red and the fox is wild". This way, the model should be able to learn
that it should output the logical AND of the selected premises if it is not able to make a deduction. Second, sometimes the
model is able to make a correct deduction but it can somehow rephrase it to make it closer to the hypothesis or in the right 
direction. For example, from a single compound premise "the dog is red and the fox is wild" the model can output "the dog is red" if 
that is the hypothesis that we are looking for. To mitigate this issue, we can append the hypothesis to the input of the 
deduction model. In this way, the model should be able to output the correct deduction that is closer to the hypothesis but
on the other hand we are risking the possibility of the model outputting the hypothesis directly when it doesn't have enough
information to make a deduction. Again, I also used a Flan-T5 model to implement the deduction module.

## Dataset

For data, I used the same dataset that was used in the NLProofs paper but I also used one single model to work on both
datasets (Proof-writer and EntailmentBank) with the hope to achieve a more generalized model at the end. I took each of
the branches of the entailment trees in those datasets and converted them to a sequence to sequence format for both selection
and deduction modules.

## Proof Search

The proof search component puts all the modules together and tries to find the proof for a given hypothesis. I began by 
implementing a greedy approach where I first select the premises using the selection by doing greedy generation and then input 
the selected premises to the deduction module to get the next step. Then we add the new premise to the context of the selector
to perform next iteration and so on. The stopping criterion is when the model outputs the hypothesis or when the model reaches
maximum number of iterations. To estimate if we've reached the stopping criterion, we use the [BLEURT model](https://github.com/google-research/bleurt)
by google which is a trained metric which estimates the similarity between two texts. We use this metric to estimate how close the
induced sentence to the hypothesis is and if it is close enough we stop the proof search.

Now, the problem with this approach is similar to any greedy approach when we are searching over an answer in a tree. We have
a score for each step of the proof given the beam scores of the selection generator. Given those scores we can build many possible 
trees and we have to choose the best one. This reduces our problem to a tree search problem. Since doing a complete search is not feasible,
we choose to implement a beam search algorithm to search over the possible trees. At each iteration we only keep the top-k trees
and continue the algorithm until we reach the stopping criterion. The figure below shows two steps of the proof search algorithm using 
3 beams.

![beam search](/homepage/images/ps-beamssearch.png)

## Evaluation and Results

I used the same evaluation metrics as the NLProofs paper which are **accuracy** and **F1** score for leaves, intermediate nodes
and deduction steps and also the overall accuracy. For this initial model up to this point here are the results on the first 
and second EntailmentBank dataset tasks:

![results](/homepage/images/sandd-results.png)

## Conclusion and Future Work

As shown above, with this simple design we were able to achieve a model that is able to perform reasoning task with high accuracy 
close to the performance of the much more complicated models like NLProofs and MetGEN. This shows the importance of designing
a simple method with clear objectives. In the future, I plan to work on improving the objective of both components as discussed
above to mitigate possible issues and improve the performance of the model. Also, as shown in Metgen paper, it is worth to
explore the possibility of using a general scoring function to score a whole generated tree instead of scoring each step and
combining them together which can result in a better performance. 



