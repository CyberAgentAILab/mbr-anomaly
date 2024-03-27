# NMT Models Setup
Use the following commands to download and extract the models.

## WMT'19 winner models
We use the following WMT'19 winner models ([Ng et al., 2019](https://arxiv.org/abs/1907.06616)) in our experiments. Details and usage instructions of the models can be found in the [fairseq/examples/translation](https://github.com/facebookresearch/fairseq/tree/main/examples/translation).

- English -> German
    ```bash
    curl -L https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz \
        --output wmt19.en-de.ensemble.tar.gz
    tar -xzf wmt19.en-de.ensemble.tar.gz
    mv wmt19.en-de.joined-dict.ensemble wmt19.en-de.ensemble
    ```
- German -> English
    ```bash
    curl -L https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz \
        --output wmt19.de-en.ensemble.tar.gz
    tar -xzf wmt19.de-en.ensemble.tar.gz
    mv wmt19.de-en.joined-dict.ensemble wmt19.de-en.ensemble
    ```
- English -> Russian
    ```bash
    curl -L https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz \
        --output wmt19.en-ru.ensemble.tar.gz
    tar -xzf wmt19.en-ru.ensemble.tar.gz
    ```
- Russian -> English
    ```bash
    curl -L https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz \
        --output wmt19.ru-en.ensemble.tar.gz
    tar -xzf wmt19.ru-en.ensemble.tar.gz
    ```
