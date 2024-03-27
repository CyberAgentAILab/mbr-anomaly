import argparse
import pandas as pd
import numpy as np

from sklearn import covariance

from utils.mbr_decoding_utils import (
    load_mbrd_by_candidate,
    load_mbrd_by_pseudo_ref,
)

def main(mbrd_prefix):
    print("Loading base dataframe...")
    cnd_df = load_mbrd_by_candidate(mbrd_prefix=mbrd_prefix, use_cache=True)
    pref_df = load_mbrd_by_pseudo_ref(mbrd_prefix=mbrd_prefix, use_cache=True)

    print("Computing Covariance matrices...")
    def compute_example_cov(example_pref_df, nodup_candidates=False):
        example_index = example_pref_df.index.get_level_values("example_index")[0]

        if not nodup_candidates:
            example_util_T = np.stack(
                example_pref_df["utilities_T"].values
            )
        else:
            example_cnd_df = cnd_df.loc[example_index]
            example_cnd_nodup_indices = example_cnd_df.index[~example_cnd_df.duplicated(subset="sentence")]
            example_util_T = np.stack(
                example_pref_df["utilities_T"].map(lambda x: x[example_cnd_nodup_indices]).values
            )
        cov = covariance.EmpiricalCovariance().fit(example_util_T)
        return cov

    covs = pref_df.groupby('example_index').apply(compute_example_cov)
    covs_nodup_candidates = pref_df.groupby('example_index').apply(
        compute_example_cov, nodup_candidates=True
    )
    covariance_result = pd.concat(
        [covs, covs_nodup_candidates],
        keys=["all_candidates", "nodup_candidates"],
        axis=1
    )

    print("Saving results...")
    covariance_result.to_pickle(mbrd_prefix + ".example_covariance.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbrd_prefix", type=str)
    args = parser.parse_args()

    main(args.mbrd_prefix)
