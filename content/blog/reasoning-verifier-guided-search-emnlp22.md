---
author: ["Navid Madani"]
title: "Enabling Reasoning Capabilities in Large Language Models - Part 1"
date: "2024-03-15"
tags: ["Reasoning"]
ShowToc: true
draft: false
TocOpen: true
---

# Introduction

In the rapidly evolving landscape of artificial intelligence, Large Language Models (LLMs) have emerged as 
frontrunners in generating text that is not only coherent but also strikingly fluent. At a glance, these models seem 
almost sentient in their ability to weave words together, a capability that has been integral to their widespread adoption in various applications,
from composing emails to drafting articles. However, the crux of their functionality raises a pertinent question:
**when LLMs are essentially designed to predict the next token in a sequence with the goal of minimizing perplexity, 
can we truly say they are capable of reasoning?**

The concept of reasoning, or the ability to make sense of information and draw logical conclusions,
seems at odds with the core operation of LLMs. This disparity brings us to an intriguing proposition:
what if LLMs were to remain the masterful writers they are, while a separate, dedicated module handles the intricate dance of thinking and reasoning?
This reasoning module could very well be another LLM, but fine-tuned to excel not in crafting sentences, but in navigating
the complex pathways of decision-making and logical inference.

In this blog post, we delve into an interesting approach that seeks to bridge this gap between generation and reasoning
in LLMs. **We'll explore the paper "Generating Natural Language Proofs with Verifier-Guided Search"** a piece of work 
that I find particularly compelling for its elegant solution to this problem. By examining this paper, we aim to shed 
light on how the field of AI is advancing towards models that don't just speak, but also think, marking a significant
leap forward in our quest to create truly intelligent systems. We will go over the code and the paper to understand the
approach and possibly use their insights for improving our own models.

Since this work was implemented using an older version on pytorch lightning, I had to clone the project in 
another github repo and make some changes to make the code work. Here is the link to the repo and the paper


<div align="center">
	<a href="https://github.com/navidmdn/llm_reasoning">
	    <img src="/homepage/images/github-mark.png" width="40" height="40" alt="Github">
	</a>
	<a href="https://arxiv.org/abs/2205.12443">
	    <img src="/homepage/images/paper.png" width="40" height="40" alt="Paper">
	</a>
</div>

# Generating Natural Language Proofs with Verifier-Guided Search

## The Idea

 this work introduces a novel method named NLProofS, which stands for Natural Language Proof Search.
 This method distinguishes itself by generating proof steps in a sequential manner, conditioned on the given hypothesis,
 thereby ensuring the relevance of each step to the overarching logical argument.
The brilliance of NLProofS lies in its integration of an independent verifier, a component trained to assess the validity
 of each proof step. This verifier plays a crucial role in the method's ability to minimize errors commonly seen in other
 models, such as the generation of illogical or irrelevant proof steps—a phenomenon often referred to as "hallucination" 
 in AI language models. By leveraging the verifier's assessments, NLProofS employs a search strategy that doesn't just
 sequentially generate proof steps but seeks out paths through the proof that are maximized for their overall validity,
 according to the verifier's judgment.

The paper reports that NLProofS sets new benchmarks in performance on challenging datasets such as **EntailmentBank and RuleTaker**,
illustrating its efficacy in generating complex, human-like proofs. The method's success is particularly notable in 
scenarios involving distractors or in cases requiring the model to sift through a large corpus to identify relevant 
supporting facts. This not only showcases the model's ability to maintain logical coherence in proofs but also 
its adeptness at handling real-world complexities where irrelevant information may exist.

To sum up, the overall task of NLProofS is to generate an entailment tree that represents a logical proof, 
where each node in the tree corresponds to an argument or a rule and the root of the tree represents the hypothesis:

![NLProofS](/homepage/images/nlproofs.png)

## Task definitions

#### Task 1 (No-distractor)
In this simplest form, the task provides the model with a hypothesis and a set of supporting facts that are precisely the leaf nodes required to construct the ground truth proof tree. There are no extra or irrelevant pieces of information (distractors), meaning each provided fact directly contributes to proving the hypothesis. This task assesses the model's ability to generate coherent proofs when all necessary information is straightforward and directly available.

#### Task 2 (Distractor)
This task increases complexity by introducing distractors alongside the supporting facts. The model is given a hypothesis and a set of 25 sentences that includes both the relevant facts needed for the proof and additional irrelevant or misleading facts (distractors). The model's challenge is to sift through this mixed information, identify the relevant facts, and construct a logical proof for the hypothesis, demonstrating its capability to focus on pertinent information amidst potential noise.

#### Task 3 (Full-corpus)
The most challenging scenario, Task 3, requires the model to engage in a form of information retrieval alongside proof generation. The model is presented with a hypothesis and a large corpus containing thousands of sentences, from which it must first identify and retrieve relevant supporting facts before generating the proof. This task simulates a more realistic scenario where the model must navigate extensive information sources, akin to real-world reasoning and research tasks, to find the facts necessary to construct a valid proof.

## Step-wise Prover

At the core of NLProofS is a model which is responsible for generating proofs step by step. The first step in building 
the whole pipeline of their proposed system is to understand how we this component works. In this work the authors experiment
with a T5 and Bart models to generate the proof steps. But first, let's understand the way they preprocess the data for
fine-tuning these models.

### Data Preparation

Starting with the original EntailmentBank or RuleTaker datasets, the following steps are taken to prepare data for training the prover module. 
We first process each example to a Proof objects defined in [`proof.py`](https://github.com/princeton-nlp/NLProofS/blob/main/prover/proof.py) file which contains
a list of `ProofStep` objects each presenting a single step in the proof. Later in `__getitem__` method of the [`StepwiseDataset`](https://github.com/princeton-nlp/NLProofS/blob/main/prover/datamodule.py) class,
the real data augmentation and filtering for building each example happens. For each of the training examples, the `get_example_train` is called. It first
shuffles the proof context which only shuffles the identifiers of proof sentences (sent1, sent2 -> sent2, sent1). Then it uses the `to_tree` method defined
inside the `Proof` class to convert the whole proof to a tree using `ete3` library. The following code is the full implementation of the `get_example_train` method at time
of writing this post:

```python 
    def get_example_train(self, ex: Example) -> Example:
        proof = ex["proof"].shuffle_context()

        # Sample the pro[of step.
        tree = proof.to_tree()
        int_node = random.choice(get_internal_nodes(tree))

        # Sample the goal.
        if self.sample_goal == "hypothesis":
            goal_node = tree.get_tree_root()
        else:
            assert self.sample_goal == "intermediates"
            ancestors = int_node.get_ancestors()
            assert int_node not in ancestors
            ancestors.append(int_node)
            goal_node = random.choice(ancestors)

        # Sample the partial proof.
        proved_subtrees = [node for node in int_node.children if not node.is_leaf()]
        if int_node is not goal_node:
            unproved_child = int_node
            for node in int_node.iter_ancestors():
                for child in node.children:
                    if child is unproved_child or child.is_leaf():
                        continue
                    if self.subtree_proved_all_or_none:
                        if random.random() < self.subtree_proved_prob:
                            proved_subtrees.append(child)
                    else:
                        proved_subtrees.extend(
                            collect_proved_subtrees(child, self.subtree_proved_prob)
                        )
                if node is goal_node:
                    break
                else:
                    unproved_child = node
        proved_subtrees.reverse()
        random.shuffle(proved_subtrees)
        partial_proof = " ".join(serialize(t) for t in proved_subtrees)

        # goal_context
        input_seq = f"$hypothesis$ = {goal_node.sent} ; $context$ = {proof.serialize_context()} ; $proof$ = {partial_proof}"

        premises = [node.name for node in int_node.children]
        random.shuffle(premises)
        output_seq = " & ".join(premises)
        if goal_node is int_node:
            output_seq = output_seq + " -> hypothesis;"
        else:
            output_seq = output_seq + f" -> int: {int_node.sent};"

        ex = deepcopy(ex)
        ex["proof"] = proof
        ex["input_seq"] = input_seq
        ex["output_seq"] = output_seq
        return ex
```

After building the tree we randomly sample an internal node of the tree and sample either an ancestor of the internal node or the hypothesis as the goal
(it depends on the training config). Then we construct the partial proof as follow:
- all the subtrees of current internal node must be in the partial proof
- from the internal node up to the goal node, we randomly sample a subtree with a probability of `subtree_proved_prob` to have some level of data augmentation
- we shuffle the subtrees and concatenate them to build the partial proof

Finally, they build the input sequences for the model. The input sequence is built as follows:
- the hypothesis is the goal node
- the context is the proof context
- the proof is the partial proof

At each stage they use the [`serialize`](https://github.com/princeton-nlp/NLProofS/blob/main/common.py) method to convert the tree to a string.

### What can be improved?

So the training procedure is a straightforward sequence to sequence training with a partial proof as input and the next proof step as output.
However, we observe some data augmentation in which the model learns to only choose relevant information from the context to continue the proof.
In my opinion we are only observing a class of positive samples here. That is, we are only learning to continue the proof from contexts in which
there are sufficient information to continue the proof. As an example, what happens if the stepwise prover is given 
two sentences `the dog is brown` and `the fox is red`? Maybe it helps the model to understand that the only logical continuation of
such context is to `AND` the two sentences. This might be a good step in preventing wrong proof steps and reducing the load on the verifier.

Another point to think about is what happens if the model is provided with context that allows multiple continuations. Since we are
augmenting the context with irrelevant subtrees, it can introduce new paths to the proof or when we are dealing with multiple leaf nodes, there can be
multiple continuations to build the tree for one step. However, we're giving them all to the model and ask it to provide a next step proof.
These duplicate labels can be confusing for the model and can lead to a model that is not able to generalize well.

