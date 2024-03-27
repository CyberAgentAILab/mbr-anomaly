# Datasets Setup
Use the following commands to download and extract the datasets in this directory.

## newstest2019
Test set for the WMT19 news translation task [[link](https://www.statmt.org/wmt19/translation-task.html)].
```bash
curl -L http://data.statmt.org/wmt19/translation-task/test.tgz --output newstest2019.tgz

mkdir newstest2019

tar -xzf newstest2019.tgz -C newstest2019
```
