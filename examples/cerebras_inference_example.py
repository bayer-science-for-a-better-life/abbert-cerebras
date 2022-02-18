from importlib import resources

from pathlib import Path
from typing import Optional

import pandas as pd

from abbert2.vendored.cerebras_bayer_shared.bert.tf.extract_features import ExtractEmbeddingsFromBert
from abbert2.vendored.cerebras_bayer_shared.bert.tf.utils import get_params


def read_cerebras_model_params(run_id: int = 15) -> dict:
    with resources.path('abbert2.vendored.cerebras_bayer_shared.bert.tf', 'configs') as path:
        return get_params(path / f'run{run_id}_params.yaml')


def find_cerebras_model_checkpoint(
    run_id: int = 15,
    checkpoint_step: Optional[int] = None,
    checkpoints_path: Path = Path.home() / 'cerebras' / 'final_trained_models'
) -> Path:

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


if __name__ == '__main__':

    run_id = 15
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    params = read_cerebras_model_params(run_id=run_id)
    checkpoint_path = str(find_cerebras_model_checkpoint(run_id=run_id))
    model = ExtractEmbeddingsFromBert(params, checkpoint_path)
    embeddings_df = pd.DataFrame(list(model.extract_embeddings(sequences)))
    embeddings_df.info()
