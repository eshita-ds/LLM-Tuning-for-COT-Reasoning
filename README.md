# Specializing Large language Models using Chain of Thought Reasoning

## Introduction
Large Language Models (LLMs) have revolutionized the present market landscape and are driving innovations due to their ability to generalize solutions to problems across multiple domains. Chain of Thought Reasoning (CoT) is used to overcome the challenge of LLM hallucinating information when dealing with inputs/prompts that require domain specific knowledge. The objective of this project is to build an Intelligent System on top of an LLM fine-tuned for CoT reasoning using reinforcement learning and prompting strategies. This system will provide responses tailored to education domain, where individual steps and question context play a key role in solving various scenarios. The data for this project has been scraped and collected from various open-source datasets.

Further, open-source foundational models within 3-9 billion parameter space like Llama 3.2, Gemma 3, Phi 3.5 Mini and DeepSeek R1 have been utilized. These models were enhanced for Chain-of-Thought reasoning using fine-tuning strategies like LoRA, reinforcement learning algorithms like DPO and GRPO and prompting strategies like Chain-of-Draft, Self-Consistency and Tree-of-Thought. The cleaning of data for model fine-tuning was orchestrated with the help of Apache Airflow using data pipeline infrastructure on AWS cloud platform. The model responses were evaluated using BLEU, ROUGE, METEOR, chrF and custom CoT g-Eval metrics like Factual Accuracy, Logical Coherence and Step Alignment.

<img width="350" height="400" alt="image" src="https://github.com/user-attachments/assets/b4da4284-1b8b-4d9f-aa37-31091805d57d" />

## Solution Overview
The aim of this project is to build an intelligent system trained on large volumes of data on specific domains that, with the aid of CoT reasoning, can break down complex problems into
steps and provide a response with clear and concise steps and reasoning. We will aid our models by implementing a RAG database along with employing techniques like Self-Consistency
training. The self-consistency training helps LLM(s) learn how to approach a problem by taking different paths and then choosing a response that is consistent across the majority of the paths. This, when used in parallel with CoT, helps boost the performance of the models across the trained domain (Wang et al., 2022). Similarly, RAG helps by including external knowledge sources that are used by LLM(s) to validate or verify the generated responses, thus improving the overall accuracy. These techniques, when combined, have the potential to make existing LLM(s) efficient on specific domains, opening avenues for widespread usage across multiple domains. Our work will help the community, including students, teachers, and researchers, to get detailed solutions with multi-step reasoning for complex reasoning problems that are often plagued with hallucinations when using standard prompting with LLM(s). This will help build up trust and enable deeper explorations into this domain. 

The project will follow a multi-step process for the Intelligent Tutoring system that will implement various techniques to complement the CoT reasoning. The system will undergo benchmarking and baselining and will be evaluated against multiple available models and industry-standard benchmarks with a specific domain focus. The project will need substantial amounts of annotated data for training and evaluating the models. Various publicly available data sources and repositories will be used for scraping the data for education domains. Data gathering will primarily include techniques like web scraping, APIs, and manual annotations. Data for the education domain will be sourced from resources like Khan Academy, Coursera, edX, MIT OpenCourseWare, SQUAD dataset, and various mathematical problem datasets like MathQA, GSM8K, SVAMP, Arith3K, TAL-SCQ5K-EN, MAWPS, and Math Olympiad problems. In addition to this, the project will use standard Python libraries like BeautifulSoup and Scrapy to scrape data from open data resources. Sources like Wikibooks, OER Commons, Khan Academy
API, Math Stack Exchange, etc. will be used for the education domain.

<table>
  <tr>
    <th><img width="640" height="450" alt="image" src="https://github.com/user-attachments/assets/d0a00bb8-cea5-4f84-b463-620bcadabdbf" /></th>
    <th><img width="640" height="300" alt="image" src="https://github.com/user-attachments/assets/afa92139-511a-4931-91e9-15aa006f74ce" /></th>
  </tr>
</table>

## Project Requirements
<table>
  <tr>
    <th>Capabilities</th>
    <td>NLP, problem-solving, reasoning, explainable answers, and continuous learning</td>
  </tr>
  <tr>
    <th>Data Pipeline</th>
    <td>Apache Airflow, AWS, Hugging face Model hub, Gradio Interface</td>
  </tr>
  <tr>
    <th>Datasets</th>
    <td>Pre-annotated data for Mathematics, Logical Reasoning</td>
  </tr>
  <tr>
    <th>Training Techniques</th>
    <td>CoT techniques with combinations of Self-Consistency Training (SCT), Chain of Draft, GRPO, DPO and Tree of thought</td>
  </tr>
  <tr>
    <th>Evaluation Metrics</th>
    <td>BLEU score, ROGUE score, Chrf, Correctness, GEval correctness, GEval logical coherence (LLM as a Judge)</td>
  </tr>
  <tr>
    <th>Benchmarks</th>
    <td>MATH, GSM8K, SVAMP</td>
  </tr>
</table>

### System Design
Intelligent Tutoring System has primarily two parts architecturally, the User Interface with a Chat Option and an LLMOps Pipeline for continuous model training and refinement. The
solution works with a baseline model that is identified through rigorous testing and performance comparisons to identify the most optimal model for chain-of-thought reasoning. Once the model is identified, the fine-tuned model is registered with Hugging Face Model Hub and that serves as the backbone model for the entire solution. The two parts with respect to system architecture are as outlined.

The user Interface for the “Intelligent Tutoring System” will allow users to login, interact and save their chat history. The web application will be publicly available and will
present user with an option to login. The credentials will be authenticated with a backend database allowing access control to the application. Once logged in, the previous chat history for the user will be fetched from the backend database allowing user to review the history of past interactions. User will have option to either continue the chat based on a previous interaction or start a new conversation. As part of a new conversation, user will have an option to type in the prompt/query and press “Enter” or click submit button the UI. This will initiate an internal chain that will fetch the registered model from the model registry and relevant context from a RAG database. The final prompt will be generated using a pre-defined prompt template and will be fed to an LLM using a LangChain QA pipeline. The entire interaction is handled while ensuring efficient user authentication and data retrieval.

<table>
  <tr>
    <td><img width="628" height="310" alt="image" src="https://github.com/user-attachments/assets/71601833-a9d1-4e1a-81a8-d7ada18a7830" /></td>
    <td><img width="412" height="607" alt="image" src="https://github.com/user-attachments/assets/6b968110-40ec-4d33-9f0e-ea1308c36fee" /> </td>
  </tr>
</table>

The core principle behind developing “Intelligent Tutoring System” is to be able to finetune LLMs to allow chain of thought reasoning and mimic human way of thinking. This can be achieved to a certain extent by retraining the LLMs with new datasets on a regular basis. The data will be generated using a feedback system that will allow users to rate the
responses provided by the LLMs. As outlined in the User Interface above, the users will have a way to capture feedback on the responses provided by the models. These responses will then get stored into a database as historical cache. The pipeline aims to utilize this historical data cache to fine-tune or retrain the baseline model to improve efficiency and performance. Similar cleaning and pre-processing steps will be performed as with the baseline training data. The cleaned data will be stored in AWS S3 bucket for easy storing and retrieval. This data will then be retrieved, preprocessed, stored as parquet files on S3 bucket. For final fine tuning, parquet files will be parsed and fed to a training pipeline. The hyperparameters and model metrics will be logged into a monitoring platform and will be compared against baseline model. If there is an improvement in performance, the baseline model in the model registry will be updated with a new versioned model. This new model will then be used as the primary model for performing inference in the user interface. The entire pipeline will be executed on pre-defined schedules for continuous training. 

<table>
  <tr>
    <td><img width="418" height="318" alt="image" src="https://github.com/user-attachments/assets/f5f752cb-d838-4b00-8b1d-62e5869171ab" /></td>
    <td><img width="412" height="397" alt="image" src="https://github.com/user-attachments/assets/2c1b2e2c-10a8-4b6c-aaa7-aa4335617ffe" /></td>
  </tr>
</table>

## Infrastructure Overview

<table>
  <tr>
    <th>Application Hosting</th>
    <td>Hugging Face Spaces: The user interface is hosted on HF Spaces with ZeroGPU access. ZeroGPU is a shared GPU concept that allows dynamic allocation of resources. This setup is a good fit for our use case as we need GPU for inference.</td>
  </tr>
  
  <tr>
    <th>Database Management</th>
    <td>AWS RDS will be used to host the MySQL database for the user interface. It will serve three primary functions: user authentication, session management, and historical data storage. AWS EC2 will host ChromaDB, which will serve as the vector store for context retrieval.</td>
  </tr>

  <tr>
    <th>Orchestration</th>
    <td>Apache Airflow, hosted on an Oracle Cloud compute instance, will orchestrate the model re-training and fine-tuning pipeline. Oracle Cloud is used instead of AWS EC2 due to resource constraints.</td>
  </tr>

  <tr>
    <th>Model Management</th>
    <td>Fine-tuned models will be stored and versioned on the Hugging Face Model Hub, and the latest model will be used within the application. MLflow, hosted on AWS EC2, will be used for model registry, experiment tracking, and comparison.</td>
  </tr>

  <tr>
    <th>Fine-Tuning</th>
    <td>Kaggle will be used to access GPUs for fine-tuning the models.</td>
  </tr>
</table>

<img width="475" height="175" alt="image" src="https://github.com/user-attachments/assets/398d96b9-f2ff-48db-9fcd-0bdbd3694041" />

## Model Comparisons
<table>
  <td><img width="588" height="528" alt="image" src="https://github.com/user-attachments/assets/85e666e6-cf1a-4310-8d6f-e20e4c2e674c" /></td>
  <td><img width="588" height="468" alt="image" src="https://github.com/user-attachments/assets/a0f2546d-96bf-4088-bcc5-bd45478e3cfe" /></td>
</table>

## Results

- The Llama 3.2 1B-DPO model achieves the highest G-eval score of 0.38378, making it the best-performing model among the models that we tested. 
- Llama 3.2 1B will be the primary model used with the Airflow pipeline owing to its strong performance and small size, making it the preferred choice for continuous integration and fine-tuning.

<img width="523" height="305" alt="image" src="https://github.com/user-attachments/assets/ac43e190-c807-4ef7-8d4f-d3b6a9fa155b" />

**Key insights:**

- DPO-based fine tuning technique shows impressive performance, particularly with the Llama 3.2 1B model.
- Combining GRPO with DPO or COD generally yields lower g-eval scores 
- Llama 3.2 3B-GRPO+COT’s performance is almost at par with DPO on a model with lower number of parameters. 
- Gemma3 1B and 4B yielded almost similar results with different techniques.
- DeepSeek-R1 with self consistency was the worst performing model.

<img width="1106" height="429" alt="image" src="https://github.com/user-attachments/assets/763bf868-1f47-4d4a-b1e8-c034d9ae39f5" />

## User Interface
<table>
  <tr>
    <td><img width="631" height="609" alt="image" src="https://github.com/user-attachments/assets/dca03e01-6422-45e7-af36-05b18dfcdacd" /></td>
    <td><img width="672" height="399" alt="image" src="https://github.com/user-attachments/assets/e9b65734-b78b-4d64-952f-0ae4ec12edcf" /></td>
  </tr>
  <tr>
    <td><img width="647" height="456" alt="image" src="https://github.com/user-attachments/assets/4e6cac5b-89fb-49dd-8a1e-89548cce9d8a" /></td>
    <td><img width="681" height="351" alt="image" src="https://github.com/user-attachments/assets/46a6f7db-5e8f-4131-a6fb-976aef5d1c1c" /></td>
  </tr>
</table>

## License

MIT License

Copyright (c) 2024 Eshita Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
