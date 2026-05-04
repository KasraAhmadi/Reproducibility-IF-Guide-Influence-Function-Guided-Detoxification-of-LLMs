## IF-Guide: Influence Function-Guided Detoxification of LLMs 

- IF-Guide: Influence Function-Guided Detoxification of LLMs

&nbsp;

----

## Installation

Create the conda environment (you can use any environment with `python>=3.10`) and install the necessary packages:

```bash
conda create -n IF-Guide python=3.10
conda activate IF-Guide
pip install -r requirements.txt
```

Next, navigate to the working directory:

```bash
cd src
```

&nbsp;

----

### 1. Fit the Inverse Hessian Approximation Factors

Run:

```bash
./scripts/fit_factors.sh
```

This calls `fit_factors.py` and takes the following primary arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to fit factors on. |
| `--checkpoint_dir`     | Path to the saved model. If using a pre-trained model, set it to `None`. |
| `--train_indices_path` | Path to the train indices used to train the model (not necessary if using the entire 
dataset). Must match the exact indices and be in the same order. We provide the indices of our one-billion-token [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) subset and set their path as the default.|
| `--output_dir`           | Path to save the Hessian approximation data. |
| `--max_train_samples`    | Maximum training samples (to limit training size) |

&nbsp;

### 2. Compute Token-Wise Scores

Run:

```bash
./scripts/compute_scores.sh
```

This runs `compute_scores.py` with the following key arguments (in addition to most of the arguments used to compute factors):

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to compute scores for. |
| `--checkpoint_dir`     | Path to the saved model. If using a pre-trained model, set it to `None`. |
| `--save_id`            | Tag appended to the end of the save directory for custom naming. |
| `--save_dir`           | Directory to save scores in (within the original factors directory).
| `--factors_path`       | Path to the directory containing the (inverse) Hessian factors fit in the previous step. |
| `--query_dataset`      | The query dataset for constructing the query gradient. Currently, the only option is `RTP`. | 
| `--toxic_query_indices_path` | Path to the indices from the query dataset pertaining to *toxic* demonstrations. We provide our toxic subset from RTP in `../data/RTP/query_indices/toxic_indices.npy`. | 
| `--nontoxic_query_indices_path` | Path to indices for *non-toxic* queries. We provide our non-toxic subset from RTP in `../data/RTP/query_indices/nontoxic_indices.npy`. |
| `--max_train_samples`    | Maximum training samples (to limit training size) |

&nbsp;

### 3. Select Influential Toxic Tokens

Run:

```bash
./scripts/build_toxic_token_mask.sh
```

This runs `build_toxic_token_mask.py`. It takes the following primary arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to build the mask for. |
| `--scores_path`     | Path to the scores computed in the prior step. |
| `--window`            | The context window length. |
| `--toxicity_threshold`           | The threshold for determining toxic tokens (as a percentile, e.g., 0.99).
| `--max_tokens`       | The maximum number of toxic tokens to select. |
| `--query_dataset`      | The query dataset for constructing the query gradient. Currently, the only option is `RTP`. | 
| `--inspection_idx` | We automatically print out the suppressed tokens for a single training example in red. This argument specifies which example to print based on its ranking (e.g., 0 is the highest-ranked training example).  | 
| `--max_train_samples`    | Maximum training samples (to limit training size) |

&nbsp;

### 4. Suppressing Toxic Tokens During Training/Fine-Tuning

After computing the toxic tokens mask for a particular model, you can specify the `--toxic_token_mask_path` and `--toxic_lambda` arguments in `./scripts/train.sh` (and `./scripts/finetune.sh`) to train/fine-tune models with IF-Guide.

```bash
./scripts/finetune.sh
```

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`             | Name of the model to train. Must be registered in `utils/registry.yaml` with a corresponding tokenizer (see existing models for examples) |
| `--save_id`                | Descriptor tag used for output directory naming.        |
| `--toxic_token_mask_path`  | Path to a token mask (generated via IF-Guide). Use `None` for standard training. |
| `--toxic_lambda`           | The strength of the penalty term used by our training objective. |
| `--max_train_samples`    | Maximum training samples (to limit training size) |

&nbsp;

----

## Evaluation

We provide code for evaluating explicit toxicity (via [Detoxify](https://github.com/unitaryai/detoxify/tree/master)), implicit toxicity (via [ToxiGen-RoBERTa](https://huggingface.co/tomh/toxigen_roberta)), and fluency (measured on [LAMBADA](https://huggingface.co/datasets/EleutherAI/lambada_openai) and [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)).

&nbsp;

### Explicit Toxicity Evaluation

Run:

```bash
./scripts/run_toxicity_eval.sh
```

This runs `run_toxicity_eval.py`, which has the following main arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to evaluate. |
| `--checkpoint_dir`     | Path to the saved model. If using a pre-trained model, set it to `None`. |
| `--dataset`            | Dataset to evaluate on. Either `RTP`, `AttaQ`, or `BOLD`. |
| `--save_dir`           | Directory to save the results in.
| `--decoding_defense`       | Decoding-time defense to apply. `none` or `rad`. Does not apply to our OpenWebText evaluation. |
| `--save_outputs`      | Whether to save the model's outputs. | 

&nbsp;

### Fluency Evaluation

Run:

```bash
./scripts/run_fluency_eval.sh
```

It has the following primary arguments:

| Argument                   | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--model_name`         | Name of the model to evaluate. |
| `--checkpoint_dir`     | Path to the saved model. If using a pre-trained model, set it to `None`. |
| `--dataset`            | Dataset to evaluate on. Either `RTP`, `AttaQ`, or `BOLD`. |
| `--save_dir`           | Directory to save the results in.
| `--decoding_defense`       | Decoding-time defense to apply. `none` or `rad`. Does not apply to our OpenWebText evaluation. |

&nbsp;

Please contact Zachary Coalson (coalsonz@oregonstate.edu) for any questions and recommendations.
