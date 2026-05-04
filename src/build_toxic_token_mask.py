"""
Script to construct the toxic token masks used by IF-Guide to 
suppress toxicity during training. Uses token-wise influence scores
computed with `compute_scores.py`.
"""
import logging
import math
import os
import numpy as np
import argparse

import torch
from kronfluence.analyzer import Analyzer
from transformers import AutoTokenizer
from colorama import Fore, init
from tqdm import tqdm

from utils.models import get_registry_config
from utils.datasets import get_tokenized_openwebtext

init()


def parse_args():
    parser = argparse.ArgumentParser(description="Build toxic token mask using token-wise influence scores.")

    # Model and datasets
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of HF model to evaluate.",
    )
    parser.add_argument(
        "--query_dataset",
        type=str,
        default="RTP",
        choices=["RTP"],
        help="Dataset used to compute the query scores."
    )
    parser.add_argument(
        "--train_indices_path",
        type=str,
        default="../data/openwebtext/train_indices.npy",
        help="Path to the training indices for the dataset.",
    )

    # Output/saving
    parser.add_argument(
        "--save_id",
        type=str,
        default="",
        help="Identifier for the save path."
    )
    parser.add_argument(
        "--inspection_idx",
        type=int,
        default=0,
        help="Index of the example to inspect."
    )

    # IF arguments
    parser.add_argument(
        "--scores_path",
        type=str,
        default=None,
        help="Path to the influence scores.",
    )
    parser.add_argument(
        "--factor_strategy",
        type=str,
        default="ekfac",
        choices=["ekfac", "identity"],
        help="Method used for computing Hessian approximation."
    )

    # Arguments for our toxic token selection algorithm
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Window size for the mask."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20_000_000,
        help="Max number of tokens to penalize."
    )
    parser.add_argument(
        "--toxicity_threshold",
        type=float,
        default=0.99,
        help="Toxicity threshold for selecting toxic tokens."
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Max number of training samples to score.",
    )

    parser.add_argument(
        "--use_tfidf",
        action="store_true",
        default=False,
        help="Apply TF-IDF weighting to influence scores before token selection.",
)

    return parser.parse_args()


def color_strength(word: str, strength: int) -> None:
    strength = max(0, min(1, strength))
    intensity = math.floor(strength * 255)
    color = f"\033[38;2;{intensity};0;0m"
    print(f"{color}{word}{Fore.RESET}", end="")


def min_max_normalize(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize a tensor to the range [0, 1].
    """
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + eps)


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the harmonic mean of two tensors.
    """
    a = a.float()
    b = b.float()
    return 2 * a * b / (a + b + eps)

def compute_document_frequencies(train_dataset, vocab_size: int) -> tuple[np.ndarray, int]:
    """
    Compute document frequencies from the training dataset.
    Returns df_array (per token) and N (total docs).
    """
    df = np.zeros(vocab_size, dtype=np.int32)
    N = len(train_dataset)
    
    for example in tqdm(train_dataset, desc="Computing document frequencies"):
        token_ids = np.array(example["input_ids"])
        unique_tokens = np.unique(token_ids)
        df[unique_tokens] += 1
    
    return df, N

def tfidf_weight_scores(
    influence_scores: torch.Tensor,   # shape (num_samples, seq_len)
    input_ids: torch.Tensor,          # shape (num_samples, seq_len)
    df_array: np.ndarray,             # shape (vocab_size,)
    N: int,
) -> torch.Tensor:
    """
    Apply TF-IDF weighting: S'_ij = S_ij * log(N / (df_j + 1))
    """
    # Get IDF for every token position
    ids_np = input_ids.cpu().numpy()                    # (B, T)
    df_vals = df_array[ids_np]                          # (B, T)
    idf_weights = np.log(N / (df_vals + 1))             # (B, T)
    idf_weights = torch.tensor(idf_weights, dtype=influence_scores.dtype, device=influence_scores.device)
    
    return influence_scores * idf_weights


def build_toxic_token_mask(
    influence_scores: torch.Tensor,   # shape (num_samples, seq_len)
    input_ids: torch.Tensor,          # shape (num_samples, seq_len) -- NEW
    df_array: np.ndarray,             # shape (vocab_size,)          -- NEW
    N: int,                           # total docs in corpus          -- NEW
    toxicity_threshold: float = 0.99,
    context_window: int = 5,
    max_tokens: int = 1_000_000,
    use_tfidf: bool = True,           # toggle for ablation           -- NEW
) -> torch.Tensor:
    """
    Builds a toxic token mask where:
      True = toxic token,
      False = non-toxic token.
    
    Ranking is done using the harmonic mean of:
      (1) the number of values above the toxicity threshold
      (2) the sum of those values
    """
    scores = influence_scores.detach()
    if use_tfidf:
        scores = tfidf_weight_scores(scores, input_ids, df_array, N)

    B, T = scores.shape

    # Compute threshold
    threshold = np.quantile(scores.cpu().float().numpy().flatten(), toxicity_threshold)

    # Compute stats per record
    above_thresh = (scores > threshold).float()
    count_above = min_max_normalize(above_thresh.sum(dim=1))         
    sum_above = min_max_normalize((scores * above_thresh).sum(dim=1))

    # Harmonic mean ranking
    harmonic_scores = harmonic_mean(count_above, sum_above)

    # Select top-k rows
    sorted_indices = torch.argsort(harmonic_scores, descending=True).tolist()

    mask = torch.zeros_like(scores, dtype=torch.bool)
    already_selected = torch.zeros_like(scores, dtype=torch.bool)
    used_tokens = 0

    pbar = tqdm(total=max_tokens, desc="Building toxic token mask", unit="tokens")
    for i in sorted_indices:
        toxic_indices = (scores[i] > threshold).nonzero(as_tuple=True)[0]
        for idx in toxic_indices:
            center = idx.item()
            start = max(0, center - context_window)
            end = min(T, center + context_window + 1)

            # Get the span that hasn't been selected yet
            new_mask = ~already_selected[i, start:end]
            new_tokens = new_mask.sum().item()

            if used_tokens + new_tokens > max_tokens:
                print(f"Reached max token limit of {max_tokens}. Stopping...")
                return mask, sorted_indices[:i]

            # Select only where needed
            mask[i, start:end][new_mask] = True
            already_selected[i, start:end][new_mask] = True

            used_tokens += new_tokens
            pbar.update(new_tokens)

    return mask, sorted_indices 


def main():
    args = parse_args()

    config = get_registry_config(args.model_name)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Building/loading toxic token mask for {args.model_name}...")

    train_indices = np.load(args.train_indices_path)
    if args.max_train_samples is not None:
        train_indices = train_indices[:args.max_train_samples]
    train_dataset = get_tokenized_openwebtext(config, train_indices)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['path'], use_fast=True, trust_remote_code=True)

    save_path = (
        f"../data/toxic_token_masks/"
        f"{args.factor_strategy}_{args.query_dataset}_{args.model_name}_kas_method"
        f"{'_' + args.save_id if args.save_id else ''}/"
        f"mask_toks={round(args.max_tokens / 1e6, 1)}m"
        f"_p={args.toxicity_threshold}"
        f"_w={args.window}.pt"
    )
    vocab_size = tokenizer.vocab_size
    df_array, N = compute_document_frequencies(train_dataset, vocab_size)

    if os.path.exists(save_path):
        mask = torch.load(save_path, map_location="cpu", weights_only=True)
    else:
        scores = Analyzer.load_file(args.scores_path)['all_modules'][0]

        # NEW: extract input_ids aligned with scores
        input_ids = torch.tensor(
            np.array([example["input_ids"] for example in train_dataset])
        )
        if args.max_train_samples is not None:
            input_ids = input_ids[:args.max_train_samples]

        mask, _ = build_toxic_token_mask(
            scores,
            input_ids=input_ids,       # NEW
            df_array=df_array,         # NEW
            N=N,                       # NEW
            toxicity_threshold=args.toxicity_threshold,
            context_window=args.window,
            max_tokens=args.max_tokens,
            use_tfidf=args.use_tfidf,            # NEW
        )
        torch.save(mask, save_path)
    
    top_indices = torch.argsort((mask == True).sum(dim=1), descending=True).tolist()
    top_idx = top_indices[args.inspection_idx]

    # Get the corresponding sequences
    logging.info("Influential example (selected toxic tokens in red):")
    words = tokenizer.batch_decode(train_dataset[top_idx]["input_ids"])
    strengths = mask[top_idx] == True

    for word, strength in zip(words, strengths):
        color_strength(word, strength)

if __name__ == "__main__":
    main()
    print("\nDone.")