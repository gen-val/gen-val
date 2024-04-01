# Are LLMs Better at Generation or Validation?

This repository explores the capabilities of Large Language Models (LLMs) in two main tasks: generation and validation. Through a series of experiments and analysis, we aim to understand the strengths and limitations of LLMs in performing these functions.

## Introduction

Large Language Models (LLMs) like GPT (Generative Pre-trained Transformer) have shown remarkable abilities in generating human-like text. However, their efficiency in validation tasks, where the goal is to verify the accuracy or reliability of given information, is less explored. This project compares the performance of LLMs on generation tasks versus validation tasks, shedding light on their potential and paving the way for more effective applications in various fields.

## Prerequisites

Before you can run the experiments in this repository, ensure you have the following installed:
- Python 3.8 or above
- Required Python packages: dill, collie-bench, etc. (A complete list is available in `requirements.txt`)

## Setup

To set up your environment to run the experiments, follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/repo_url
```

2. Navigate to the cloned directory:

```bash
cd gen_disc
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

To compare the generation and validation capabilities across all datasets from the paper, you can run the following script. This script sets up the environment and initiates a series of tests using a predefined log probabilities setting.

```bash
sh src/run_FRQA_to_MCQA.sh
```

Ensure you are in the project's root directory before running the script.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Anonymized during paper review