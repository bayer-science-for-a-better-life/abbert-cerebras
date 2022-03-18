import time
from importlib import resources

from pathlib import Path
from typing import Optional, List, Dict, Union, Iterable

import pandas as pd

from abbert2.vendored.cerebras_bayer_shared.bert.tf.extract_features import ExtractEmbeddingsFromBert
from abbert2.vendored.cerebras_bayer_shared.bert.tf.utils import get_params


def read_cerebras_model_params(run_id: int = 15) -> dict:
    with resources.files('abbert2.vendored.cerebras_bayer_shared.bert.tf') as path:
        return get_params(path / 'configs' / f'run{run_id}_params.yaml')


def find_cerebras_model_checkpoint(
    run_id: int = 15,
    checkpoint_step: Optional[int] = None,
    checkpoints_path: Path = Path.home() / 'cerebras' / 'final_trained_models'
) -> Path:

    # cerebras uses estimator checkpoint utils for this

    # find the concrete checkpoints path
    candidates = sorted(checkpoints_path.glob(f'run{run_id}*'))
    if len(candidates) > 1:
        raise ValueError(f'Too many candidates for run_id={run_id}: {candidates}')
    if len(candidates) == 0:
        raise ValueError(f'Cannot find a path to run_id={run_id} in {checkpoints_path}')
    checkpoint_path = candidates[0]

    # use the last step if no checkpoint_id is passed
    if checkpoint_step is None:
        checkpoint_candidates = sorted(checkpoint_path.glob('model.ckpt-*.index'),
                                       key=lambda x: int(str(x.stem).split('-')[1]))
        if 0 == len(checkpoint_candidates):
            raise ValueError(f'Cannot find any saved checkpoint in {checkpoint_path}')
        checkpoint_path = checkpoint_candidates[-1].with_suffix('')
    else:
        checkpoint_path = checkpoint_path / f'model.ckpt-{checkpoint_step}'
        if not checkpoint_path.is_file():
            raise ValueError(f'Cannot find checkpoint {checkpoint_path}')

    return checkpoint_path


class InfiniteAbbertCerebras(ExtractEmbeddingsFromBert):

    def __init__(self, run_id=15, remove_cls=True, remove_sep=True, warmup=False):
        super().__init__(
            params=read_cerebras_model_params(run_id=run_id),
            checkpoint_path=str(find_cerebras_model_checkpoint(run_id=run_id)),
            args=None
        )
        self.remove_cls = remove_cls
        self.remove_sep = remove_sep
        self._embedder_cache = None
        self._next = None
        if warmup:
            self.warmup()

    # --- Iterator with len 0 to trick

    def __len__(self):
        # trick clunky cerebras code
        return 0

    def __iter__(self):
        # trick clunky cerebras code
        return self

    def __next__(self):
        # trick clunky cerebras code
        if self._next is None:
            raise StopIteration
        next_element = self._next
        self._next = None
        return next_element

    # --- Lifecycle

    def warmup(self):
        self('A')

    def _embedder(self):
        if self._embedder_cache is None:
            self._embedder_cache = super().extract_embeddings(self)
        return self._embedder_cache

    def __del__(self):
        try:
            next(self._embedder())       # Raise Stop Iteration
            self._embedder_cache = None  # TF session die
        except AttributeError:
            ...

    # --- Simple API sequence -> embeddings dict

    def __call__(self, sequences: Union[str, Iterable[str]]) -> List[Dict]:
        if isinstance(sequences, str):
            sequences = sequences,
        embeddings = []
        for sequence in sequences:
            self._next = sequence
            embedding = {'sequence': sequence}
            start = 0 if not self.remove_cls else 1
            end = start + len(sequence) + 1 if not self.remove_sep else start + len(sequence)
            embedding.update({
                tensor_name: tensor[start:end]
                for tensor_name, tensor in next(self._embedder()).items()}
            )
            embeddings.append(embedding)
        return embeddings


if __name__ == '__main__':

    run_id = 15
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    start = time.perf_counter()
    embedder = InfiniteAbbertCerebras(run_id=run_id, warmup=True)
    print(f'Setting-up Taken: {time.perf_counter() - start:.2f}s')

    start = time.perf_counter()
    embeddings_df = pd.DataFrame(embedder(sequences))
    print(f'Inference Taken {time.perf_counter() - start:.2f}s')

    embeddings_df.info()
    #
    # Data columns (total 15 columns):
    #  #   Column                   Non-Null Count  Dtype
    # ---  ------                   --------------  -----
    #  0   sequence                 3 non-null      object
    #  1   features/input_ids       3 non-null      object
    #  2   features/input_mask      3 non-null      object
    #  3   encoder_layer_0_output   3 non-null      object
    #  4   encoder_layer_1_output   3 non-null      object
    #  5   encoder_layer_2_output   3 non-null      object
    #  6   encoder_layer_3_output   3 non-null      object
    #  7   encoder_layer_4_output   3 non-null      object
    #  8   encoder_layer_5_output   3 non-null      object
    #  9   encoder_layer_6_output   3 non-null      object
    #  10  encoder_layer_7_output   3 non-null      object
    #  11  encoder_layer_8_output   3 non-null      object
    #  12  encoder_layer_9_output   3 non-null      object
    #  13  encoder_layer_10_output  3 non-null      object
    #  14  encoder_layer_11_output  3 non-null      object
    #

# --- Notes

#
# Originally, we have maxlength to be 160 or 180, and these raw output tensors as numpy arrays like:
# features/input_ids (182)
# features/input_mask (182 => L+1 0s, (182 - L+1) 1s)
# encoder_layer_0_output (182 x D)
# encoder_layer_1_output (182 x D)
# encoder_layer_2_output (182 x D)
# ...
# encoder_layer_(L-1)_output (182 x D)
#
