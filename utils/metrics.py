from typing import List, Dict, Any, Tuple, Optional
from functools import partial, wraps
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def wrapped_partial(f, *args, **kwargs):
    return wraps(f)(partial(f, *args, **kwargs))



def build_metric_fn(
    metric_name: str,
    comet_dir: str = None,
    comet_model: str = "wmt20-comet-da",
    comet_bsize: int = 256,
    bleurt_dir: str = None,
    n_cpus=1,
    n_gpus=1,
    devices: List[int] = None,
    num_workers: Optional[int] = None,
    progress_bar: bool = True,
    only_sentence_level: bool = True,
):
    if metric_name in ["comet20", "comet22"]:
        assert comet_dir is not None
        from comet import download_model, load_from_checkpoint

        # download comet and load
        comet_path = download_model(comet_model, comet_dir)
        comet_model = load_from_checkpoint(comet_path)

        return partial(
            comet,
            comet_model=comet_model,
            comet_bsize=comet_bsize,
            n_gpus=n_gpus,
            devices=devices,
            num_workers=num_workers,
            progress_bar=progress_bar,
        )

    elif metric_name == "bleurt":
        assert bleurt_dir is not None
        from bleurt import score

        bleurt_scorer = score.LengthBatchingBleurtScorer(bleurt_dir)

        return partial(bleurt, bleurt_scorer=bleurt_scorer)

    elif metric_name == "bleu":
        return partial(
            bleu,
            progress_bar=progress_bar,
            parallel=n_cpus,
            only_sentence_level=only_sentence_level,
        )

def comet(
    hyps: List[str],
    refs: List[str],
    srcs: List[str],
    comet_model: object,
    comet_bsize: int,
    n_gpus: int = 1,
    devices: List[int] = None,
    num_workers: Optional[int] = None,
    progress_bar: bool = True,
):
    comet_input = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
    ]
    # sentence-level and corpus-level COMET
    comet_output = comet_model.predict(
        comet_input,
        batch_size=comet_bsize,
        gpus=n_gpus,
        devices=devices,
        # sort_by_mtlen=True,
        num_workers=num_workers,
        progress_bar=progress_bar,
    )
    return comet_output.scores, comet_output.system_score

def bleurt(
    hyps: List[str],
    refs: List[str],
    bleurt_scorer: object,
    srcs: List[str] = None,
    bleurt_bsize: str = 64,
):
    bleurt_scores = bleurt_scorer.score(
        references=refs,
        candidates=hyps,
        batch_size=bleurt_bsize,
    )
    assert type(bleurt_scores) == list
    return bleurt_scores, np.array(bleurt_scores).mean()

def bleu(
    hyps: List[str],
    refs: List[str],
    srcs: List[str] = None,
    progress_bar: bool = True,
    parallel: int = 1,
    only_sentence_level: bool = True,
    pre_dispatch: str = "16*n_jobs",
):
    import sacrebleu

    bleu_fn = lambda *args: sacrebleu.sentence_bleu(*args).score

    iterator = (
        delayed(bleu_fn)(hyp, [ref]) if parallel > 1 else bleu_fn(hyp, [ref])
        for hyp, ref in zip(hyps, refs)
    )

    if parallel > 1 and progress_bar:
        iterator = ProgressParallel(
            total=len(hyps),
            n_jobs=parallel,
            batch_size=50000,
            pre_dispatch=pre_dispatch,
        )(iterator)
    elif progress_bar:
        iterator = tqdm(iterator, total=len(hyps))

    sentence_scores = list(iterator)

    corpus_score = None
    if not only_sentence_level:
        corpus_score = sacrebleu.corpus_bleu(hyps, [refs]).score

    return sentence_scores, corpus_score
