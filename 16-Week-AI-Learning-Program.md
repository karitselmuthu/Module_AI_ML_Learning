# Applied Machine Learning, Generative AI, and LLM Engineering

**Duration:** 16 weeks  
**Suggested workload:** 10–12 hours per week  
**Format:** Concept lessons, guided coding, practical labs, and project development  
**Entry requirements:** Basic Python, familiarity with functions and data structures, and introductory algebra and probability.

## How we will use this program

Use this program as the primary learning plan. Keep the learner's existing detailed topic breakdowns as backup reference material. After completing each module, use those breakdowns as a review checklist to identify gaps and reinforce learning.

The existing module outline is sufficient to begin. Additional breakdowns can be incorporated during review without delaying the learning program.

The suggested timing, tools, and assessment weights below are planning choices and can be adjusted to the learner's progress and available resources. Use one evolving project, with a practical milestone at the end of each module.

## Program outcomes

By the end of the program, learners should be able to:

- Prepare data, train machine learning models, and evaluate their performance.
- Build and improve neural networks using Python frameworks.
- Explain and apply core generative modeling concepts.
- Develop LLM applications using retrieval, tools, and structured workflows.
- Combine text, image, video, and audio processing in an application.
- Deploy and monitor an AI application with repeatable evaluation and cost controls.

## Weekly learning format

- **Concepts and demonstrations:** 2 hours.
- **Guided practical lab:** 3 hours.
- **Independent project work:** 4–5 hours.
- **Review and assessment:** 1–2 hours.

## Module 1: Fundamentals of Machine Learning with Practical Applications

### Weeks 1–3

**Objective:** Build a reliable baseline model and understand the complete machine learning workflow.

| Week | Topics | Practical work | Deliverable |
|---|---|---|---|
| 1 | Supervised and unsupervised learning; problem framing; Python data tools; exploratory analysis | Load, clean, and visualize a dataset using NumPy, pandas, and plotting libraries | Data exploration notebook and problem statement |
| 2 | Regression, classification, preprocessing, feature engineering, and train/validation/test splits | Train baseline models using scikit-learn; build a preprocessing pipeline | Reproducible training pipeline |
| 3 | Cross-validation, hyperparameter tuning, classification and regression metrics, clustering, and error analysis | Compare models, inspect failures, and present results | Machine learning application milestone |

**Project milestone:** Build a ticket classifier, demand predictor, or churn prediction application.

**Assessment:** Explain metric selection, demonstrate prevention of data leakage, and compare the final model against a simple baseline.

**Review:** Compare completed learning against the learner's detailed Module 1 breakdown and revisit any gaps.

## Module 2: Deep Learning Technologies with Practical Python Tools and Frameworks

### Weeks 4–6

**Objective:** Train, evaluate, and improve a neural network.

| Week | Topics | Practical work | Deliverable |
|---|---|---|---|
| 4 | Tensors, neural networks, activation functions, loss functions, backpropagation, and optimization | Implement a training loop using PyTorch | Working neural network and training report |
| 5 | Convolutional networks, transfer learning, regularization, and data augmentation | Adapt a pretrained image model to a small labeled dataset | Image classification application |
| 6 | Embeddings, sequence modeling, attention, and transformer fundamentals | Compare model behavior, inspect errors, and package inference | Deep learning application milestone |

**Project milestone:** Add image categorization or document-image classification to the evolving application.

**Assessment:** Interpret training curves, explain overfitting, and justify model and training choices.

**Review:** Compare completed learning against the learner's detailed Module 2 breakdown and revisit any gaps.

## Module 3: Generative AI Technologies with Python Tools and Frameworks

### Weeks 7–8

**Objective:** Understand generative modeling and build a controlled generation workflow.

| Week | Topics | Practical work | Deliverable |
|---|---|---|---|
| 7 | Generative versus discriminative models; latent spaces; autoencoders; variational autoencoders | Train a small VAE on a compact image dataset | Notebook demonstrating reconstruction and sampling |
| 8 | Diffusion concepts, conditioning, sampling, and generation evaluation | Use a pretrained diffusion model to generate images and compare outputs across settings | Generative AI application milestone |

**Project milestone:** Create a visual asset generator with saved prompts, configuration, and an output review process.

**Assessment:** Explain the difference between reconstruction and generation, and document quality limitations and data-use considerations.

**Scope:** Training a diffusion model from scratch is an optional extension.

**Review:** Compare completed learning against the learner's detailed Module 3 breakdown and revisit any gaps.

## Module 4: LLMs, Fine-Tuning, Agents, and RAG

### Weeks 9–10

**Objective:** Build a document-grounded assistant and evaluate when retrieval, tools, or fine-tuning are appropriate.

| Week | Topics | Practical work | Deliverable |
|---|---|---|---|
| 9 | Tokens, context windows, transformer language models, prompting, structured outputs, embeddings, chunking, and retrieval | Build a document ingestion and retrieval pipeline | RAG assistant with source references |
| 10 | Fine-tuning fundamentals; parameter-efficient adaptation; tool calling; agent workflows; LangChain | Run a small adaptation experiment and add a constrained tool workflow | Evaluated LLM and RAG application milestone |

**Project milestone:** Build a knowledge assistant that answers from uploaded documents, provides references, and handles insufficient evidence.

**Assessment:** Evaluate retrieval quality and answer grounding separately; compare prompting, retrieval, and fine-tuning for a defined task.

**Scope:** Fine-tuning is a small guided experiment; a larger training run is optional and depends on available compute.

**Review:** Compare completed learning against the learner's detailed Module 4 breakdown and revisit any gaps.

## Module 5: Applications with Text, Video, Image, and Audio

### Weeks 11–12

**Objective:** Process multiple input formats through a consistent application interface.

| Week | Topics | Practical work | Deliverable |
|---|---|---|---|
| 11 | Text extraction, summarization, image understanding, OCR, video frame sampling, and timestamp handling | Build text and image workflows; summarize a short video using sampled frames and a transcript | Text, image, and video processing features |
| 12 | Speech recognition, speech synthesis, transcript processing, and multimodal evaluation | Add audio transcription and spoken output; test noisy or incomplete inputs | Multimodal application milestone |

**Project milestone:** Extend the assistant to accept documents, images, and audio, with a short-video summarization workflow.

**Assessment:** Measure task accuracy, processing time, and failure behavior for each input type.

**Review:** Compare completed learning against the learner's detailed Module 5 breakdown and revisit any gaps.

## Module 6: Building Integrated Generative AI Applications

### Weeks 13–14

**Objective:** Combine the earlier components into one usable product.

| Week | Topics | Practical work | Deliverable |
|---|---|---|---|
| 13 | Application architecture, task routing, shared state, API design, and user experience | Connect retrieval, classification, and generation components | Integrated application prototype |
| 14 | Model configuration, workflow customization, caching, fallbacks, access controls, and input validation | Add error handling and run end-to-end evaluations | Integrated generative AI application milestone |

**Project milestone:** Deliver a working application with a user interface, documented workflows, and a repeatable evaluation set.

**Assessment:** Demonstrate complete user journeys, including unsupported requests, missing documents, and component failures.

**Review:** Compare completed learning against the learner's detailed Module 6 breakdown and revisit any gaps.

## Module 7: Productionizing Applications—End-to-End Pipeline

### Weeks 15–16

**Objective:** Deploy the application and establish a practical operating process.

| Week | Topics | Practical work | Deliverable |
|---|---|---|---|
| 15 | Serving models; application packaging; containers; secrets; automated checks; deployment environments | Package the application and deploy it to a chosen hosting environment | Accessible deployment and deployment guide |
| 16 | Logging, monitoring, latency, cost tracking, versioning, regression evaluation, and rollback | Run load and failure checks; demonstrate a release and rollback process | Final application, operating guide, and presentation |

**Assessment:** Demonstrate deployment reproducibility, trace a failed request, explain operating costs, and recover from a faulty release.

**Review:** Compare completed learning against the learner's detailed Module 7 breakdown, including MLOps practices, CI/CD, model maintenance, serving options, Docker, and Kubernetes. Use this review to identify topics that need additional study or an extension lab.

## Suggested Python toolkit

Use one primary tool per responsibility to keep the learning workload manageable. This is the toolkit proposed in the initial plan; check current compatibility and suitability when preparing each teaching environment.

| Responsibility | Suggested tools |
|---|---|
| Data preparation and analysis | NumPy, pandas |
| Visualization | Matplotlib, Seaborn |
| Traditional machine learning | scikit-learn |
| Deep learning | PyTorch |
| Pretrained models and adaptation | Transformers, PEFT |
| Diffusion workflows | Diffusers |
| Retrieval storage | FAISS or Chroma |
| LLM workflow orchestration | LangChain |
| Application API | FastAPI |
| Demonstration interface | Streamlit |
| Testing and packaging | pytest, Docker |
| Experiment tracking | MLflow |

Pin package versions when preparing the teaching environment and provide a tested environment file for each module.

## Evolving capstone: Multimodal Knowledge and Support Assistant

**Purpose:** Help users find information, categorize requests, and understand uploaded content.

The project develops through seven milestones:

1. **Machine learning:** Categorize incoming support requests.
2. **Deep learning:** Classify attached images.
3. **Generative AI:** Add a visual generation workflow.
4. **LLMs and RAG:** Answer questions using a curated document collection.
5. **Multimodal processing:** Accept images, audio, and short videos.
6. **Integration:** Combine features in one interface with reliable error handling.
7. **Deployment:** Publish the application with monitoring, evaluation, and operating documentation.

The visual generation feature can remain a separate module demonstration if it does not serve the final product's users.

## Assessment structure

| Component | Weight |
|---|---:|
| Weekly practical labs | 25% |
| Concept checks | 10% |
| Module milestone demonstrations | 25% |
| Final integrated application | 30% |
| Documentation and final presentation | 10% |
| **Total** | **100%** |

Evaluate project work on correctness, evaluation quality, reproducibility, usability, and the learner's ability to explain tradeoffs.

## Final submission requirements

- Source code with setup and execution instructions.
- Data sources and preparation notes.
- Trained model artifacts or instructions for obtaining them.
- Evaluation dataset, metrics, and error analysis.
- Working application demonstration.
- Architecture diagram and deployment instructions.
- Estimated operating costs and known limitations.
- Monitoring and recovery guide.
- A short presentation covering the problem, solution, results, and next improvements.

## Preparation and resource planning

Before Week 1, provide a Python refresher, environment setup guide, sample datasets, and a short readiness exercise.

Arrange GPU access for deep learning and generative modeling labs. Set a shared spending limit for paid model services, and provide smaller datasets and lightweight exercises for learners with limited compute.

## Module review checklist

After each module:

- [ ] Explain the main concepts without relying on notes.
- [ ] Complete the practical exercises and milestone.
- [ ] Record evaluation results and inspect failures.
- [ ] Compare the completed work with the detailed backup breakdown.
- [ ] Record missing topics and complete targeted follow-up study.
- [ ] Save notes, code, and a short reflection before proceeding.
