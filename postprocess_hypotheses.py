import os
import argparse
import json
import random
from typing import List, Dict, Any
import multiprocessing as standard_mp

import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as torch_mp

from utils.metrics import build_metric_fn

def read_generation_output(generation_output: str, generated_data: dict):
    """
    Read and parse the generation output of fairseq-generate and store the data in a dictionary.
    """
    # Read the generation output
    generation_name = os.path.basename(generation_output)
    for line in open(generation_output, "r"):
        if not line.startswith(("S-", "T-", "H-", "D-", "P-")):
            continue

        id_, *data = line.strip().split("\t")
        data_type, idx = id_.split("-")

        generated_data[idx] = generated_data.get(idx, {})
        if data_type == "S":
            # Source sentence
            assert len(data) == 1
            source = data[0]
            if "source" in generated_data[idx]:
                assert generated_data[idx]["source"] == source, \
                    f"Source sentence for sentence {idx} is not consistent across generations"
            else:
                generated_data[idx]["source"] = source
        elif data_type == "T":
            # Reference translation
            assert len(data) == 1
            reference = data[0]
            if "reference" in generated_data[idx]:
                assert generated_data[idx]["reference"] == reference, \
                    f"Reference translation for sentence {idx} is not consistent across generations"
            else:
                generated_data[idx]["reference"] = reference
        elif data_type == "H":
            # Hypothesis
            pass
        elif data_type == "D":
            # Detokenized hypothesis
            assert len(data) == 2
            _, hypothesis = data
            hypotheses = generated_data[idx].get("hypotheses", [])
            hypotheses.append({
                "generation_name": generation_name,
                "sentence": hypothesis,
            })
            generated_data[idx]["hypotheses"] = hypotheses
        elif data_type == "P":
            # Token probabilities
            assert len(data) == 1
            assert idx in generated_data
            logprobs = [float(x) for x in data[0].split()]
            generated_data[idx]["hypotheses"][-1]["sum_logprobs"] = np.sum(logprobs)
            generated_data[idx]["hypotheses"][-1]["mean_logprobs"] = np.mean(logprobs)

    # Sanity check
    num_samples = len(generated_data[idx]["hypotheses"])
    for idx in generated_data:
        assert len(generated_data[idx]["hypotheses"]) == num_samples, \
            f"Number of hypotheses for sentence {idx} is not {num_samples}"

    for example_index in generated_data:
        for hypothesis_index, hypothesis in enumerate(generated_data[example_index]["hypotheses"]):
            generated_data[example_index]["hypotheses"][hypothesis_index] = {
                "hypothesis_index": hypothesis_index,
                **hypothesis,
            }
    return generated_data

def _compute_metric(
        rank: int,
        metric_name,
        args: argparse.Namespace,
        indices_by_rank: Dict[int, List[int]],
        result_by_rank: Dict[int, Any],
        flat_hypotheses: List[str],
        flat_references: List[str],
        flat_sources: List[str],
):
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
        metric_name=metric_name,
        comet_model=args.comet_model,
        comet_dir=args.comet_dir,
        comet_bsize=args.comet_bsize,
        bleurt_dir=args.bleurt_dir,
        n_cpus=args.n_cpus,
        n_gpus=args.n_gpus,
        devices=devices,
        num_workers=num_workers,
        only_sentence_level=False,
    )

    flat_scores, _ = metric_fn(
        hyps=[flat_hypotheses[i] for i in indices_to_process],
        refs=[flat_references[i] for i in indices_to_process],
        srcs=[flat_sources[i] for i in indices_to_process] if flat_sources is not None else None,
    )

    result_by_rank[rank] = flat_scores

def main(args):
    # Read the generation output
    if os.path.exists(args.result_prefix + ".hypotheses.json"):
        print("Loading existing hypotheses file...")
        with open(args.result_prefix + ".hypotheses.json", "r") as f:
            hypotheses_data = json.load(f)
    else:
        print("Reading the generation output...")
        generated_data = read_generation_output(generation_output=args.generation_output,
                                                generated_data={})

        num_examples = len(generated_data) if args.num_examples < 0 else args.num_examples
        random.seed(args.seed)
        example_indices = sorted(
            random.sample(list(generated_data.keys()), num_examples),
            key=int
        )
        print(f"Picking {len(example_indices)} examples: {example_indices}")
        hypotheses_data = {idx: generated_data[idx] for idx in example_indices}

        os.makedirs(os.path.dirname(args.result_prefix), exist_ok=True)
        with open(args.result_prefix + ".hypotheses.json", "w") as f:
            json.dump(hypotheses_data, f, indent=4, ensure_ascii=False)

    result_fpath = args.result_prefix + ".scores.pkl"
    if os.path.exists(result_fpath):
        print("Loading existing result file...")
        result_df = pd.read_pickle(result_fpath)
    else:
        result_data = []
        for example_index in hypotheses_data:
            for hypothesis_index in range(len(hypotheses_data[example_index]["hypotheses"])):
                result_data.append({
                    "example_index": example_index,
                    "hypothesis_index": hypothesis_index,
                })
        result_df = pd.DataFrame(result_data).set_index(["example_index", "hypothesis_index"])

    print("Evaluating all the hypotheses...")
    for metric_name in args.eval_metrics:
        if f"{metric_name}_score" in result_df.columns and args.resume:
            print(f"Skipping {metric_name} because it is already computed.")
            continue

        example_and_hypothesis_indices = []
        flat_hypotheses = []
        flat_references = []
        flat_sources = []
        for example_index in hypotheses_data:
            num_hypotheses = len(hypotheses_data[example_index]["hypotheses"])

            example_and_hypothesis_indices += [
                (example_index, hypothesis_index) for hypothesis_index in range(num_hypotheses)
            ]
            flat_hypotheses += [x["sentence"] for x in hypotheses_data[example_index]["hypotheses"]]
            flat_references += [hypotheses_data[example_index]["reference"]] * num_hypotheses
            flat_sources += [hypotheses_data[example_index]["source"]] * num_hypotheses

        num_hyps = len(flat_hypotheses)
        indices_by_process = {
            rank : indices.tolist() for rank, indices in enumerate(
                np.array_split(np.arange(num_hyps), args.world_size)
            )
        }

        if args.world_size > 1:
            print(f"Spawning {args.world_size} processes...")
            manager = standard_mp.Manager()
            result_by_rank = manager.dict()
            torch_mp.spawn(
                fn=_compute_metric,
                args=(
                    metric_name,
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
            _compute_metric(
                rank=0,
                metric_name=metric_name,
                args=args,
                indices_by_rank=indices_by_process,
                result_by_rank=result_by_rank,
                flat_hypotheses=flat_hypotheses,
                flat_references=flat_references,
                flat_sources=flat_sources,
            )

        # Gather the results from all processes
        flat_scores = np.concatenate(
            [result_by_rank[rank] for rank in range(args.world_size)],
        )

        example_and_hypothesis_indices = pd.MultiIndex.from_tuples(
            example_and_hypothesis_indices, names=["example_index", "hypothesis_index"]
        )
        result_df.loc[example_and_hypothesis_indices, f"{metric_name}_score"] = flat_scores

        # Save the result at each metric evaluation
        result_df.to_pickle(result_fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-output",
        type=str,
        required=True,
        help=("File containing all source, target, and hypothesis sentences and log probabilities, "
              "generated by fairseq-generate"),
    )
    parser.add_argument(
        "--result-prefix",
        type=str,
        required=True,
        help="File prefix to write the results to."
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=-1,
        help="Number of examples to consider. If -1, consider all examples."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the existing result file."
    )
    parser.add_argument(
        "--eval-metrics",
        default=["comet22"],
        choices=["bleu", "comet20", "comet22", "bleurt"],
        help="Metric(s) to evaluate the chosen hypothesis",
        nargs="+",
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

    for metric_name in args.eval_metrics:
        if metric_name.startswith("comet"):
            args.comet_model = {"comet20": "wmt20-comet-da",
                                "comet22": "Unbabel/wmt22-comet-da"}[metric_name]
            break
    else:
        args.comet_model = None

    main(args)
