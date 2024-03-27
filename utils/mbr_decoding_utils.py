import re
import os
import json
import pandas as pd
import numpy as np

def load_hypotheses(hypotheses_prefix):
    hypothesis_data = json.load(open(hypotheses_prefix + ".hypotheses.json"))
    hypothesis_data_ = []
    for example_index in hypothesis_data:
        for hypothesis in hypothesis_data[example_index]["hypotheses"]:
            hypothesis_index = hypothesis.pop("hypothesis_index")
            hypothesis_data_.append({
                "example_index": example_index,
                "hypothesis_index": hypothesis_index,
                "source": hypothesis_data[example_index]["source"],
                "reference": hypothesis_data[example_index]["reference"],
                **hypothesis
            })
    hypothesis_data = pd.DataFrame(hypothesis_data_).set_index(
        ["example_index", "hypothesis_index"]
    )

    scores = pd.read_pickle(hypotheses_prefix + ".scores.pkl")

    hypothesis_data = hypothesis_data.join(scores)
    return hypothesis_data.reset_index()

def parse_hyp_prefix(prefix):
    prefix = os.path.basename(os.path.dirname(prefix))
    lang, sampling_id = prefix.split(".")[-2:]
    sampling_method, nbest, seed = re.split(r"nb|seed", sampling_id)
    return lang, sampling_method, int(nbest), int(seed)


def load_mbrd_by_candidate(mbrd_prefix, use_cache=False):
    if use_cache and os.path.exists(mbrd_prefix + ".candidates_cache.pkl"):
        # print("Loading candidates dataframe from preprocessed pickle")
        mbrd_data = pd.read_pickle(mbrd_prefix + ".candidates_cache.pkl")
        return mbrd_data

    print("Building candidates dataframe from data.json, scores.pkl, and utilities.pkl")

    mbrd_data = json.load(open(mbrd_prefix + ".data.json"))
    mbrd_data_ = []
    for example_index in mbrd_data:
        for candidate_index, candidate in enumerate(mbrd_data[example_index]["candidates"]):
            assert candidate["hypothesis_index"] == candidate_index, \
                (f"candidate_index ({candidate_index}) != hypothesis_index "
                 f"({candidate['hypothesis_index']})")
            mbrd_data_.append({
                "example_index": example_index,
                "candidate_index": candidate_index,
                "source": mbrd_data[example_index]["source"],
                "reference": mbrd_data[example_index]["reference"],
                **candidate,
            })
    mbrd_data = pd.DataFrame(mbrd_data_).set_index(["example_index", "candidate_index"])

    args = json.load(open(mbrd_prefix + ".args.json"))
    hypotheses_scores = pd.read_pickle(args["hypotheses_prefix_for_candidates"] + ".scores.pkl")
    hypotheses_scores.index.rename({"hypothesis_index": "candidate_index"}, inplace=True)
    mbrd_data = mbrd_data.join(hypotheses_scores)

    candidates_utilities = pd.read_pickle(mbrd_prefix + ".utilities.pkl")
    mbrd_data = mbrd_data.join(candidates_utilities)

    mbrd_data["is_mbr"] = False
    mbr_candidate_indices = mbrd_data.groupby("example_index")["expected_utility"].idxmax()
    mbrd_data.loc[mbr_candidate_indices, "is_mbr"] = True

    mbrd_data["is_top_mean_logprob"] = False
    top_mean_logprob_candidate_indices = mbrd_data.groupby("example_index")["mean_logprobs"].idxmax()
    mbrd_data.loc[top_mean_logprob_candidate_indices, "is_top_mean_logprob"] = True

    mbrd_data["is_top_sum_logprob"] = False
    top_sum_logprob_candidate_indices = mbrd_data.groupby("example_index")["sum_logprobs"].idxmax()
    mbrd_data.loc[top_sum_logprob_candidate_indices, "is_top_sum_logprob"] = True

    mbrd_data.to_pickle(mbrd_prefix + ".candidates_cache.pkl")

    return mbrd_data

def load_mbrd_by_pseudo_ref(mbrd_prefix, use_cache=False):
    if use_cache and os.path.exists(mbrd_prefix + ".pseudo_refs_cache.pkl"):
        # print("Loading pseudo_refs dataframe from preprocessed pickle")
        mbrd_data = pd.read_pickle(mbrd_prefix + ".pseudo_refs_cache.pkl")
        return mbrd_data

    print("Building pseudo_refs dataframe from data.json and scores.pkl")

    mbrd_data = json.load(open(mbrd_prefix + ".data.json"))
    mbrd_data_ = []
    for example_index in mbrd_data:
        for pseudo_ref_index, pseudo_ref in enumerate(mbrd_data[example_index]["pseudo_refs"]):
            mbrd_data_.append({
                "example_index": example_index,
                "pseudo_ref_index": pseudo_ref_index,
                "source": mbrd_data[example_index]["source"],
                "reference": mbrd_data[example_index]["reference"],
                **pseudo_ref,
            })
    mbrd_data = pd.DataFrame(mbrd_data_).set_index(["example_index", "pseudo_ref_index"])

    args = json.load(open(mbrd_prefix + ".args.json"))

    # COMET score for reference
    hypotheses_scores = pd.read_pickle(args["hypotheses_prefix_for_pseudo_refs"] + ".scores.pkl")
    hypotheses_scores.index.rename({"hypothesis_index": "pseudo_ref_index"}, inplace=True)
    mbrd_data = mbrd_data.join(hypotheses_scores)

    # Pairwise BLEU scores for all hypotheses
    if os.path.exists(args["hypotheses_prefix_for_pseudo_refs"] + ".pairwise_bleus.pkl"):
        hypotheses_pairwise_bleus = pd.read_pickle(args["hypotheses_prefix_for_pseudo_refs"] + ".pairwise_bleus.pkl")
        hypotheses_pairwise_bleus.index.rename({"hypothesis_index": "pseudo_ref_index"}, inplace=True)
        mbrd_data = mbrd_data.join(hypotheses_pairwise_bleus)

    # Utility matrices for all candidates
    utilities = pd.read_pickle(mbrd_prefix + ".utilities.pkl")
    num_examples = utilities.index.get_level_values("example_index").nunique()
    num_pseudo_refs = utilities.iloc[0]["utilities"].shape[0]
    utility_matrices = utilities["utilities"].groupby(level="example_index").agg(
        func=lambda x: np.stack(x.values)
    )
    utility_matrices_T = utility_matrices.map(lambda x: list(x.T)).to_frame()
    utility_matrices_T.rename(columns={"utilities": "utilities_T"}, inplace=True)
    utility_matrices_T["pseudo_ref_index"] = [np.arange(num_pseudo_refs)] * num_examples
    utilities_T = utility_matrices_T.explode(
        ["pseudo_ref_index", "utilities_T"]
    ).set_index("pseudo_ref_index", append=True)
    mbrd_data = mbrd_data.join(utilities_T)

    mbrd_data.to_pickle(mbrd_prefix + ".pseudo_refs_cache.pkl")
    return mbrd_data

def parse_mbrd_prefix(prefix):
    prefix = os.path.basename(os.path.dirname(prefix))
    lang, cnd_sampling_id, pref_sampling_id = prefix.split(".")[-3:]
    cnd_sampling_method, cnd_nbest, cnd_seed = re.split(r"nb|seed", cnd_sampling_id)
    pref_sampling_method, pref_nbest, pref_seed = re.split(r"nb|seed", pref_sampling_id)

    if (cnd_sampling_method, cnd_nbest) == (pref_sampling_method, pref_nbest) and (cnd_seed != pref_seed):
        seed_combination_suffix = "_diff_seeds"
    else:
        seed_combination_suffix = ""

    return (lang,
            cnd_sampling_method, int(cnd_nbest), int(cnd_seed),
            pref_sampling_method, int(pref_nbest), int(pref_seed),
            seed_combination_suffix)
