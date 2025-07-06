# llm_timeseries_synth
 This repository corresponds to the implementation for the research work titled "Evaluating the Usefulness of Large Language Models for Human Activity Recognition Data Augmentation via Few-shot Samples".


## Repository Structure

- `generating1/` and `generating2/`:  
  Code for data generation using Coral, provided in two different versions.

- `gemini/`:  
  Code for data generation using Gemini.

**These folders include prompt files and require the user to provide their own API key.**

- `datasets0/`:  
  Contains the original datasets used in our experiments.

- `utils/`:  
  Utility functions used across various modules.

- `eval/`:  
  Code for evaluating the generated data.

- `visualize/`:  
  Scripts for visualizing datasets and results.

- `preprocessing/`:  
  Code for preprocessing the datasets before generation or evaluation.

- `supplementary/`:  
  Additional supplementary material related to the study.
