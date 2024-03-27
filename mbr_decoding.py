import os
import json
import random
from argparse import ArgumentParser, Namespace
from typing import List, Dict, Any, Tuple, Optional, Union
import multiprocessing as standard_mp

import pandas as pd
import numpy as np

import torch
import torch.multiprocessing as torch_mp

from utils.metrics import build_metric_fn

def _compute_utilities_for_rank(
        rank: int,
        args: Namespace,
        indices_by_rank: Dict[int, List[int]],
        result_by_rank: Dict[int, Any],
        flat_hypotheses: List[str],
        flat_references: List[str],
        flat_sources: List[str] = None,
) -> None:
    """
    For a given rank (i.e., process id), compute the utility matrix for the given hypotheses 
    and references.
    Args:
        rank: Rank of the process.
        args: Namespace containing the arguments.
        indices_by_rank: Dictionary mapping rank to indices of examples to process.
        result_by_rank: Dictionary to store the results.
    """
    if args.world_size > 1:
        # Check args
        assert args.n_cpus == 1, "n_cpus must be 1 for multiprocessing"
        assert args.n_gpus == 1, "n_gpus must be 1 for multiprocessing"

        torch.cuda.set_device(rank)

        devices = [rank]
        num_workers = 0
    else:
        devices = None
        num_workers = None

    indices_to_process = indices_by_rank[rank]

    metric_fn = build_metric_fn(
        args.metric,
        comet_model=args.comet_model,
        comet_dir=args.comet_dir,
        comet_bsize=args.comet_bsize,
        bleurt_dir=args.bleurt_dir,
        n_cpus=args.n_cpus,
        n_gpus=args.n_gpus,
        devices=devices,
        num_workers=num_workers,
    )
    flat_utilities, _ = metric_fn(
        hyps=[flat_hypotheses[i] for i in indices_to_process],
        refs=[flat_references[i] for i in indices_to_process],
        srcs=[flat_sources[i] for i in indices_to_process] if flat_sources is not None else None,
    )

    result_by_rank[rank] = flat_utilities

def compute_utilities(
        args: Namespace,
        flat_hypotheses: List[str],
        flat_references: List[str],
        flat_sources: List[str] = None,
) -> np.ndarray:
    """
    Compute the utility matrix for the given hypotheses and references.
    Args:
        args: Namespace containing the arguments.
        flat_hypotheses: List of hypotheses. e.g., ["hypothesis 1", "hypothesis 2", ...]
        flat_references: List of references. e.g., ["reference 1", "reference 2", ...]
        flat_sources: List of source sentences. e.g., ["source 1", "source 2", ...]
    Returns:
        flat_utilities: Flattened utility matrix. e.g., [utility 1, utility 2, ...]
    """
    indices_by_process = {
        rank : indices.tolist() for rank, indices in enumerate(
            np.array_split(np.arange(len(flat_hypotheses)), args.world_size)
        )
    }
    if args.world_size > 1:
        print(f"Spawning {args.world_size} processes...")
        manager = standard_mp.Manager()
        result_by_rank = manager.dict()
        torch_mp.spawn(
            fn=_compute_utilities_for_rank,
            args=(
                args,
                indices_by_process,
                result_by_rank,
                flat_hypotheses,
                flat_references,
                flat_sources,
            ),
            nprocs=args.world_size,
            join=True,
        )
    else:
        print("Computing utility matrix in a single process...")
        result_by_rank = {}
        _compute_utilities_for_rank(
            rank=0,
            args=args,
            indices_by_rank=indices_by_process,
            result_by_rank=result_by_rank,
            flat_hypotheses=flat_hypotheses,
            flat_references=flat_references,
            flat_sources=flat_sources,
        )

    # Gather the results from all processes
    flat_utilities = np.concatenate(
        [result_by_rank[rank] for rank in range(args.world_size)],
    )

    return flat_utilities

def main(args):
    # Read the generation output
    print("Reading the generation output...")
    hypotheses_for_candidates = json.load(
        open(args.hypotheses_prefix_for_candidates + ".hypotheses.json", "r")
    )
    hypotheses_for_pseudo_refs = json.load(
        open(args.hypotheses_prefix_for_pseudo_refs + ".hypotheses.json", "r")
    )

    assert set(hypotheses_for_candidates.keys()) == set(hypotheses_for_pseudo_refs.keys()), \
        ("Translation examples for candidates and pseudo_refs are not the same."
         f" Only in candidates: {set(hypotheses_for_candidates.keys()) - set(hypotheses_for_pseudo_refs.keys())}"
         f" Only in pseudo_refs: {set(hypotheses_for_pseudo_refs.keys()) - set(hypotheses_for_candidates.keys())}")

    # Pick a subset of examples
    example_indices = sorted(list(hypotheses_for_candidates.keys()), key=int)
    num_examples = len(example_indices) if args.num_examples < 0 else args.num_examples
    random.seed(args.seed)
    example_indices = sorted(
        random.sample(example_indices, num_examples),
        key=int
    )
    print(f"Picking {num_examples} examples: {example_indices}")

    # Select the top-k candidates and pseudo_refs for each source sentence
    print((f"Selecting the top-k ({args.num_candidates}) candidates and pseudo_refs "
           f"({args.num_pseudo_refs}) for each source sentence..."))
    num_candidates = len(hypotheses_for_candidates[example_indices[0]]["hypotheses"])
    num_pseudo_refs = len(hypotheses_for_pseudo_refs[example_indices[0]]["hypotheses"])
    if num_candidates < args.num_candidates:
        raise ValueError((f"Number of candidates ({num_candidates}) is less than the "
                          f"number of candidates to select ({args.num_candidates})"))
    if num_pseudo_refs < args.num_pseudo_refs:
        raise ValueError((f"Number of pseudo_refs ({num_pseudo_refs}) is less than the "
                          f"number of pseudo_refs to select ({args.num_pseudo_refs})"))

    # Prepare the data for mbr decoding
    data_for_mbrd = {}
    for example_index in example_indices:
        data_for_mbrd[example_index] = {
            "source": hypotheses_for_candidates[example_index]["source"],
            "reference": hypotheses_for_candidates[example_index]["reference"],
            "candidates": hypotheses_for_candidates[example_index]["hypotheses"][:args.num_candidates],
            "pseudo_refs": hypotheses_for_pseudo_refs[example_index]["hypotheses"][:args.num_pseudo_refs],
        }

    os.makedirs(os.path.dirname(args.result_prefix), exist_ok=True)
    # Save args
    args.example_indices = example_indices
    with open(args.result_prefix + ".args.json", "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
    # Save the data for mbrd
    with open(args.result_prefix + ".data.json", "w") as f:
        json.dump(data_for_mbrd, f, indent=4, ensure_ascii=False)

    print("Computing utility matrices...")
    candidates_data_for_mbrd = []
    pseudo_refs_data_for_mbrd = []
    for example_index, example in data_for_mbrd.items():
        for candidate in example["candidates"]:
            candidates_data_for_mbrd.append({
                "example_index": example_index,
                "candidate_index": candidate["hypothesis_index"],
                "source_sentence": example["source"],
                "candidate_sentence": candidate["sentence"],
            })
        for pseudo_ref in example["pseudo_refs"]:
            pseudo_refs_data_for_mbrd.append({
                "example_index": example_index,
                "pseudo_ref_index": pseudo_ref["hypothesis_index"],
                "pseudo_ref_sentence": pseudo_ref["sentence"],
            })
    candidates_df_for_mbrd = pd.DataFrame(candidates_data_for_mbrd).set_index("example_index")
    pseudo_refs_df_for_mbrd = pd.DataFrame(pseudo_refs_data_for_mbrd).set_index("example_index")

    df_for_mbrd = candidates_df_for_mbrd.merge(
        pseudo_refs_df_for_mbrd,
        left_index=True, right_index=True
    ).set_index(["candidate_index", "pseudo_ref_index"], append=True)

    assert df_for_mbrd.shape[0] == num_examples * args.num_candidates * args.num_pseudo_refs, \
        (f"Number of examples ({num_examples}) * candidates ({args.num_candidates}) * "
         f"pseudo_refs ({args.num_pseudo_refs}) != rows in the dataframe ({df_for_mbrd.shape[0]})")

    flat_utilities = compute_utilities(
        args=args,
        flat_hypotheses=df_for_mbrd["candidate_sentence"].to_list(),
        flat_references=df_for_mbrd["pseudo_ref_sentence"].to_list(),
        flat_sources=df_for_mbrd["source_sentence"].to_list()
    )
    df_for_mbrd["utilities"] = flat_utilities
    result_df = df_for_mbrd["utilities"].groupby(["example_index", "candidate_index"]).apply(np.array).to_frame()
    result_df["expected_utility"] = result_df["utilities"].map(lambda x: x.mean())

    # Save the results
    result_df.to_pickle(args.result_prefix+".utilities.pkl")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hypotheses-prefix-for-candidates",
        type=str,
        required=True,
        help=("A file containing all source, target, and hypothesis sentences and log "
              "probabilities, generated by fairseq-generate. Used for candidates."),
    )
    parser.add_argument(
        "--hypotheses-prefix-for-pseudo-refs",
        type=str,
        required=True,
        help=("A file containing all source, target, and hypothesis sentences and log "
              "probabilities, generated by fairseq-generate. Used for Monte Carlo samples."),
    )
    parser.add_argument(
        "--result-prefix",
        type=str,
        required=True,
        help="File to write the results to."
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=-1,
        help="Number of examples to consider. If -1, consider all examples."
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=100,
        help="Number of candidates to consider for each source sentence."
    )
    parser.add_argument(
        "--num-pseudo-refs",
        type=int,
        default=100,
        help="Number of Monte Carlo samples to compute expected utility."
    )

    parser.add_argument(
        "--metric",
        default="comet22",
        choices=["bleu", "comet20", "comet22", "bleurt"],
        help=("Metric to use. Currently only bleu, comet and bleurt are supported. "
              "Check `qaware_decode/metrics.py` for more details."),
    )
    parser.add_argument(
        "--comet-dir",
        default=".cache/qaware_decode/comet",
        help="Directory containing the comet models.",
    )
    parser.add_argument(
        "--comet-bsize", default=200, type=int, help="batch size for gpu-based metrics"
    )
    parser.add_argument(
        "--bleurt-dir",
        default=".cache/qaware_decode/bleurt",
        help="Directory containing the bleurt models.",
    )
    parser.add_argument(
        "--n-cpus",
        default=1,
        type=int,
        help="number of cpus to use for cpu based metrics",
    )
    parser.add_argument(
        "--n-gpus",
        default=1,
        type=int,
        help="number of gpus to use for gpu based metrics",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting source sentences."
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes to use for multiprocessing",
    )
    args = parser.parse_args()

    if args.metric in ["comet20", "comet22"]:
        args.comet_model = {"comet20": "wmt20-comet-da",
                            "comet22": "Unbabel/wmt22-comet-da"}[args.metric]
    else:
        args.comet_model = None

    main(args)
