# Usage: bash preprocess_newstest2019.sh <source_lang> <target_lang>
set -e

if [ $# -ne 2 ]; then
    echo "Usage: bash $0 <source_lang> <target_lang>"
    exit 1
fi
source_lang=$1
target_lang=$2

# 1. Check language
valid_langs="en de ru"
if ! echo $valid_langs | grep -wq $source_lang; then
    echo "source_lang $source_lang is invalid"
    exit 1
fi
if ! echo $valid_langs | grep -wq $target_lang; then
    echo "target_lang $target_lang is invalid"
    exit 1
fi
if [ $source_lang == $target_lang ]; then
    echo "source_lang and target_lang are the same"
    exit 1
fi
echo "lang check passed: $source_lang -> $target_lang"

# 2. Set paths for the model, dataset, and tokenized data
model_dpath="models/wmt19.${source_lang}-${target_lang}.ensemble"
dataset_prefix="datasets/newstest2019/sgm/newstest2019-${source_lang}${target_lang}"
tokenized_data_prefix="datasets/newstest2019/tokenized/newstest2019-${source_lang}${target_lang}"
data_bin_prefix="data-bin/newstest2019/${source_lang}${target_lang}"

# 3. Tokenize the data
# 3-1. source language data
python tokenize_sgm.py \
    ${model_dpath} \
    --source-lang ${source_lang} \
    --target-lang ${target_lang} \
    --tokenizer moses \
    --bpe fastbpe \
    --bpe-codes ${model_dpath}/bpecodes \
    --sgm_fpath ${dataset_prefix}-src.${source_lang}.sgm \
    --output_fpath ${tokenized_data_prefix}.${source_lang}

# 3-2. target language data
python tokenize_sgm.py \
    ${model_dpath} \
    --source-lang ${target_lang} \
    --target-lang ${source_lang} \
    --tokenizer moses \
    --bpe fastbpe \
    --bpe-codes ${model_dpath}/bpecodes \
    --sgm_fpath ${dataset_prefix}-ref.${target_lang}.sgm \
    --output_fpath ${tokenized_data_prefix}.${target_lang}

# 4. Preprocess with fairseq-preprocess
# This will create a new directory ${data_bin_prefix}
fairseq-preprocess \
    --source-lang ${source_lang} \
    --target-lang ${target_lang} \
    --testpref ${tokenized_data_prefix} \
    --destdir ${data_bin_prefix} \
    --srcdict ${model_dpath}/dict.${source_lang}.txt \
    --tgtdict ${model_dpath}/dict.${target_lang}.txt
