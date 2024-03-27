import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import neighbors

from utils.mbr_decoding_utils import (
    load_mbrd_by_pseudo_ref,
)

def main(mbrd_prefix, n_neighbors_list):
    print("Loading base dataframe...")
    pref_df = load_mbrd_by_pseudo_ref(mbrd_prefix=mbrd_prefix, use_cache=True)

    print("Computing LOF...")
    def compute_example_lof(example_pref_df, n_neighbors):
        utilities_T_pref = np.stack(
            example_pref_df["utilities_T"].values
        )
        lof = neighbors.LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(utilities_T_pref)
        return lof

    lof_results = []
    for nn in tqdm(n_neighbors_list, desc="Computing LOF of different n_neighbors"):
        lof_result = pref_df.groupby("example_index").apply(compute_example_lof, n_neighbors=nn)
        lof_result.name = "lof_model"
        lof_result = lof_result.reset_index()
        lof_result["n_neighbors"] = nn
        lof_results.append(lof_result)
    lof_results = pd.concat(lof_results).set_index(["example_index", "n_neighbors"]).sort_index()

    print("Saving LOF results...")
    lof_results.to_pickle(f"{mbrd_prefix}.example_lof.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbrd_prefix", type=str)
    parser.add_argument("--n_neighbors_list", type=int, nargs="+",
                        default=[5, 25, 50, 75, 100])
    args = parser.parse_args()

    main(args.mbrd_prefix, args.n_neighbors_list)
