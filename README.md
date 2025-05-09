# Repository for IEEE Healthcom 2025 Submission  

This repository contains the implementation of our full paper submitted to the IEEE Healthcom 2025.  

**Title:** Efficient Fine-Tuning and Evaluation of Large Language Models for Interpreting Pediatric Hypertension Guidelines  
**Authors:** Mohamed Yaseen Jabarulla, Steffen Oeltze-Jafra, Philipp Beerbaum, Theodor Uden  
**Affiliation:** Peter L. Reichertz Institute for Medical Informatics, TU Braunschweig & Hannover Medical School  

# Overview of the Model Fine-Tuning and Deployment Process
![Image1](https://github.com/yaseen28/hypert-ai/blob/main/Screenshots/Overview.png)

# Description
------------------------------------------------------------------------------
1. We fine-tune the LLaMA 3.1 base model using the Parameter-Efficient Fine-Tuning (PEFT) based fine-tuning methods— Low-Rank Adaptation (LoRA), Quantized LoRA (QLoRA), and Bias-terms Fine-tuning (BitFit) [Code](https://github.com/yaseen28/hypert-ai/blob/main/Main_Finetune_Unsloth.ipynb) on a custom pediatric hypertension dataset [Source](https://github.com/yaseen28/hypert-ai/tree/main/Dataset)
2. The Streamlit user interface [CODE](https://github.com/yaseen28/hypert-ai/blob/main/streamlit_UI.py) allows users to choose from four quantized Language Model Models (LLMs) to access ESC consensus on pediatric hypertension.
4. In our clinical use case, we assessed each model's performance by interpreting the hypertension in children and adolescents ESC consensus PDF document. [Source](https://academic.oup.com/eurheartj/article/43/35/3290/6633855)<br/>
3. Evaluation involved using a benchmark dataset crafted by a pediatric specialist with four years of experience in pediatric cardiology manually generated twenty questions and corresponding responses by meticulously reviewing the pediatric hypertension consensu document.  [Dataset](https://github.com/yaseen28/hypert-ai/tree/main/Benchmark_Dataset).
4. Evaluated models' based on qualitative and quantitative analysis using Translation Edit Rate (TER) and BERT-based similarity metrics. Demo scripts for evaluation can be seen in the main fine tune notebook [Ref Cell Demo Scripts](https://github.com/yaseen28/hypert-ai/blob/main/Main_Finetune_Unsloth.ipynb). You can evaluate by saving model responses and ground truth in JSON format, then applying your chosen evaluation metrics.

# User Interface
A Streamlit-Powered Chat Tool for inferencing the four Large Language Models with benchmark queries.

|![Image1](https://github.com/yaseen28/hypert-ai/blob/main/Screenshots/Interface.png?raw=true) | ![Image2](https://github.com/yaseen28/hypert-ai/blob/main/Screenshots/Interface2.png?raw=true) |
|:---:|:---:|

# Evaluation Overview

![Image1](https://github.com/yaseen28/hypert-ai/blob/main/Screenshots/Evaluation.png?raw=true)
