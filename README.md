# Analyzing MBR decoding via Anomaly Detection Approach

This repository contains the code to reproduce the analysis results presented in our NAACL 2024 paper:

On the True Distribution Approximation of Minimum Bayes-Risk Decoding [[Paper]()]

## Requirements
- Python 3.9+

```bash
# Install required packages
# Specify the torch version appropriate for your environment in `requirements.txt`
pip install -r requirements.txt

# Install fastBPE here
pip install fastBPE
```

## 1 Reproduce with pre-computed results

In this section, we reproduce the analysis results presented in our paper using pre-computed translation results and utility matrices that we have already generated.

In our experiments, we used four translation pairs: English (`en`) <-> German (`de`) and English <-> Russian (`ru`), and the dataset used is newstest2019 from [WMT19](https://www.statmt.org/wmt19/translation-task.html). The NMT model used is the WMT19 winner model ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)).

### 1.1 Prepare generated hypotheses

Follow the steps below to download and extract `hypotheses`:

```bash
curl -O https://storage.googleapis.com/ailab-public/mbr-anomaly/hypotheses.tar.gz
tar -zxvf hypotheses.tar.gz
```

The `hypotheses` directory contains the results of translations generated under each setting, saved in separate directories. The naming convention for each directory is as follows:

```bash
hypotheses/wmt19.newstest2019.<source_lang>-<target_lang>.<sampling_method>nb<nbest>seed<seed>
```

- `<source_lang>`: The source language in the translation
- `<target_lang>`: The target language in the translation
- `<sampling_method>`: The generation method. For example, the following abbreviations are used:
    - `bm100`: beam search (beam size = 100)
    - `ep002`: epsilon sampling (epsilon = 0.02)
    - `tp09`: nucleus sampling (p = 0.9)
- `<nbest>`: The number of n-best for generation. Typically set to 100, matching the number of candidates for MBR decoding and the number of pseudo-references
- `<seed>`: The seed value used during generation. Either 1, 2, or 3

Each directory contains the source sentences, translated sentences, and their likelihoods for each instance saved as `e-1.hypotheses.json`. The COMET22 ([Rei et al., 2020](https://aclanthology.org/2022.wmt-1.52/)) scores for each hypothesis are stored in a DataFrame format in `e-1.scores.pkl`.

- `e-1.hypotheses.json`
    ```json
    {
        "0": {
            "source": "Welsh AMs worried about 'looking like muppets'",
            "reference": "Walisische Ageordnete sorgen sich \"wie Dödel auszusehen\"",
            "hypotheses": [
                {
                    "hypothesis_index": 0,
                    "generation_name": "wmt19.newstest2019.en-de.tp09nb100.out",
                    "sentence": "Walisische AMs besorgt, \"wie Muppets auszusehen\"",
                    "sum_logprobs": -10.9131,
                    "mean_logprobs": -0.68206875
                },
                ...
            ]
        },
        ...
    }
    ```
- `e-1.scores.pkl`
    ```python
    >>> pd.read_pickle("hypotheses/wmt19.newstest2019.en-de.tp09nb100seed1/e-1.scores.pkl")
                                    comet22_score
    example_index hypothesis_index               
    0             0                         0.700
                  1                         0.712
                  2                         0.646
    ...                                       ...
    1996          97                        0.697
                  98                        0.818
                  99                        0.547
    ```

### 1.2 Compute utility matrix for MBR decoding

Follow the steps below to download and extract `mbrd_output`:

```bash
curl -O https://storage.googleapis.com/ailab-public/mbr-anomaly/mbrd_output.tar.gz
tar -zxvf mbrd_output.tar.gz
```

The `mbrd_output` directory contains the MBR decoding results calculated for each combination of candidate and pseudo-reference. The data for candidates and pseudo-references are selected from the `hypotheses` prepared in [1.1 Prepare generated hypotheses](#11-prepare-generated-hypotheses). The naming convention for each directory is as follows:

```bash
mbrd_output/wmt19.newstest2019.<source_lang>-<target_lang>.<candidate_id>.<pseudo_ref_id>
```

- `<source_lang>`: The source language in the translation
- `<target_lang>`: The target language in the translation
- `<candidate_id>` and `<pseudo_ref_id>`: Correspond to `<sampling_method>nb<nbest>seed<seed>` from [1.1 Prepare generated hypotheses](#11-prepare-generated-hypotheses)

Each directory contains:

- `c100p100.e1000.data.json`: Data for 1,000 randomly selected instances from the dataset, recording 100 sentences for candidates and 100 sentences for pseudo-references
    ```json
    {
        "0": {
            "source": "Welsh AMs worried about 'looking like muppets'",
            "reference": "Walisische Ageordnete sorgen sich \"wie Dödel auszusehen\"",
            "candidates": [
                {
                    "hypothesis_index": 0,
                    "generation_name": "wmt19.newstest2019.en-de.ep002nb100.out",
                    "sentence": "Walisische AMs besorgt darüber, \"wie Muppets auszusehen\"",
                    "sum_logprobs": -9.650299999999998,
                    "mean_logprobs": -0.5676647058823528
                },
                ...
            ],
            "pseudo_refs": [
                {
                    "hypothesis_index": 0,
                    "generation_name": "wmt19.newstest2019.en-de.tp09nb100.out",
                    "sentence": "Walisische AMs besorgt, \"wie Muppets auszusehen\"",
                    "sum_logprobs": -10.9131,
                    "mean_logprobs": -0.68206875
                },
                ...
            ]
        },
        ...
    }
    ```

- `c100p100.e1000.utilities.pkl`: COMET22 scores recorded for each candidate against all pseudo-references for each instance.
    ```python
    >>> pd.read_pickle("hypotheses/wmt19.newstest2019.en-de.tp09/e-1.scores.pkl")
                                                      utilities  expected_utility
    example_index candidate_index
    0             0                [0.93, 0.91, 0.87, 0.88, ...              0.82
                  1                [0.85, 0.77, 0.89, 0.91, ...              0.70
                  2                [0.85, 0.77, 0.89, 0.91, ...              0.70
    ...                                                     ...               ...
    998           97               [0.88, 0.89, 0.87, 0.86, ...              0.85
                  98               [0.79, 0.79, 0.79, 0.78, ...              0.79
                  99               [0.86, 0.86, 0.85, 0.86, ...              0.85
    ```


### 1.3 Analysis of MBR decoding performance

The experimental scripts for Section 3 and Section 5 in our paper are compiled in `experiments.ipynb`. By following the steps in `experiments.ipynb`, you can reproduce our experimental results using the prepared hypotheses and utility matrix.

Regarding Anomaly Detection distances, the covariance matrix for Mahalanobis distance and the computation for Local Outlier Factor (LOF) are time-consuming, thus they have been computed in advance.

The computation results are saved in `mbrd_output/*/c100p100.e1000.example_covariance.pkl` and `mbrd_output/*/c100p100.e1000.example_lof.pkl`.

In `experiments.ipynb`, these files can be loaded later for analysis using Mahalanobis distance and LOF.

## 2 Reproduce from scratch

This section explains, with concrete examples, how to use scripts to perform calculations (e.g., generation of translation hypotheses or computation of utility matrices) that were skipped in Reproduce with pre-computed results.

### 2.1 Prepare generated hypotheses

Here, we prepare the hypotheses data as in [1.1 Prepare generated hypotheses](#11-prepare-generated-hypotheses). Hypotheses can be generated in the following four steps:

#### 2.1.1 Download dataset and model

Follow the instructions in [[datasets/README.md](datasets/README.md)] and [[models/README.md](models/README.md)] to download and unpack newstest2019 and the NMT models for each language pair.

#### 2.1.2 Preprocess dataset

Use the `preprocess_newstest2019.sh` script to tokenize and apply BPE to each language pair data of newstest2019.

For example, to preprocess the en → de pair data:

```bash
# Usage: bash preprocess_newstest2019.sh <source_lang> <target_lang>
bash preprocess_newstest2019.sh en de
```

Preprocessed data is saved in `data-bin/newstest2019/ende`.

#### 2.1.3 Generate translations

Use the preprocessed data and NMT model to generate translations. The `fairseq-generate` command is used for generation.

For example, to translate the en -> de pair using nucleus sampling (p=0.9):

```bash
data_bin_prefix="data-bin/newstest2019/ende"
model_dpath="models/wmt19.en-de.ensemble"

fairseq-generate \
    ${data_bin_prefix} \
    --path ${model_dpath}/model4.pt \
    --source-lang en \
    --target-lang de \
    --tokenizer moses \
    --bpe fastbpe \
    --bpe-codes ${model_dpath}/bpecodes \
    --batch-size 2 \
    --sampling \
    --sampling-topp 0.9 \
    --temperature 1.0 \
    --beam 100 \
    --nbest 100 \
    --seed 1 \
    > wmt19.newstest2019.en-de.tp09nb100seed1.out
```

#### 2.1.4 Postprocess generated translations

Use `postprocess_hypotheses.py` to postprocess the generated translations. At this time, the quality of each hypothesis is evaluated by specified utility functions.

For example, to evaluate and postprocess `wmt19.newstest2019.en-de.tp09nb100seed1.out` created above with COMET22:

```bash
python postprocess_hypotheses.py \
    --generation-output wmt19.newstest2019.en-de.tp09nb100seed1.out \
    --result-prefix hypotheses/wmt19.newstest2019.en-de.tp09nb100seed1/e-1 \
    --eval-metrics comet22 \
    --world-size 8
```

This creates the directory `hypotheses/wmt19.newstest2019.en-de.tp09nb100seed1` containing `e-1.hypotheses.json` and `e-1.scores.pkl`.

### 2.2 Compute utility matrix for MBR decoding

Use `mbr_decoding.py` to compute the utility matrix for MBR decoding from the hypotheses created in [2.1 Prepare generated hypotheses](#21-prepare-generated-hypotheses).

For example, if using COMET22 as the utility function, and epsilon-sampling (`ep002nb100seed1`) and nucleus-sampling (`tp09nb100seed1`) generated hypotheses as the candidate set and pseudo-reference set, respectively:

```bash
python mbr_decoding.py \
    --hypotheses-prefix-for-candidates hypotheses/wmt19.newstest2019.en-de.ep002nb100seed1/e-1 \
    --hypotheses-prefix-for-pseudo-refs hypotheses/wmt19.newstest2019.en-de.tp09nb100seed1/e-1 \
    --result-prefix mbrd_output/wmt19.newstest2019.en-de.ep002nb100seed1.tp09nb100seed1/c100p100.e1000 \
    --num-examples 1000 \
    --num-candidates 100 \
    --num-pseudo-refs 100 \
    --metric comet22 \
    --world-size 8
```

This creates the directory `mbrd_output/wmt19.newstest2019.en-de.ep002nb100seed1.tp09nb100seed1` containing `c100p100.e1000.data.json` and `c100p100.e1000.utilities.pkl`.

### 2.3 Analysis of MBR decoding performance

Before running `experiments.ipynb`, compute the covariance matrix for Mahalanobis distance and LOF scores as follows:

#### 2.3.1 Covariance matrix

Use the `compute_covariance.py` script to compute the covariance matrix from the utility matrix calculated in [2.2 Compute utility matrix for MBR decoding](#22-compute-utility-matrix-for-mbr-decoding) as follows:

```bash
python compute_covariance.py \
    --mbrd_prefix mbrd_output/wmt19.newstest2019.en-de.ep002nb100seed1.tp09nb100seed1/c100p100.e1000
```

The calculated covariance matrix is saved under the `--mbrd_prefix` directory as `c100p100.e1000.example_covariance.pkl`.

#### 2.3.2 LOF

Use the `compute_lof.py` script to compute the LOF object (sklearn.neighbors.LocalOutlierFactor) fitted on the utility matrix data calculated in [2.2 Compute utility matrix for MBR decoding](#22-compute-utility-matrix-for-mbr-decoding):

```bash
python compute_lof.py \
    --mbrd_prefix mbrd_output/wmt19.newstest2019.en-de.ep002nb100seed1.tp09nb100seed1/c100p100.e1000
```

The calculated LOF object is saved under the `--mbrd_prefix` directory as `c100p100.e1000.example_lof.pkl`.

## Citation
```bibtex
```
