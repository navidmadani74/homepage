---
author: ["Navid Madani"]
title: "Enabling Reasoning Capabilities in Large Language Models - Part 2"
date: "2024-03-15"
tags: ["Reasoning"]
ShowToc: true
draft: true
TocOpen: true
---

# Recap

In [part 1](https://navidmdn.github.io/homepage/blog/reasoning-verifier-guided-search-emnlp22/) of this series, we talked
about the verifier-guided search method which was introduced in [this paper](https://arxiv.org/abs/2205.12443). We
discussed different components of their proposed method (prover, verifier and the search algorithm) and how they work and
in the end we talked about possible design flaws and rooms for improvement. In this part, I want to design my own method
for reasoning and try to overcome the limitations of the previous method.

To be more specific, #todo:

# Design
todo:
overal design choices

## Selection Module

The selection module is responsible for learning the task of selecting feasible forward reasoning premises. When we want
to solve the problem ourselves using forward reasoning, we usually have a set of premises and we select the most relevant
ones to combine together while have an eye on the final goal. With this approach in mind we can design a selection module
that takes current state of the reasoning process and the final goal and selects the most relevant premises to combine.
Note that the selection module only selects the premises and does not combine them. The combination of the premises is
done in a separate module. We try to isolate the tasks as much as possible to make the reasoning process more modular. This
way we'll be able to evaluate and debug each of the modules more effectively.

## Deduction Module

The deduction module is responsible for combining the selected premises together to form a new premise. I tried to train
this module in a way that is independent of selection and the hypothesis that we want to prove. In the most general case,
it takes a set of premises and tries to deduct a new premise from them.

## Proof Search