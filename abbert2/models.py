"""Wrapper over our trained antibody BERTs in the cerebras CS2."""
import os
from importlib import resources
from pathlib import Path
from typing import Optional, List, Dict, Union, Iterable

from abbert2.vendored.cerebras_bayer_shared.bert.tf.extract_features import ExtractEmbeddingsFromBert
from abbert2.vendored.cerebras_bayer_shared.bert.tf.utils import get_params


# --- Paths

def _find_first_existing_directory(candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = Path(candidate)
        try:
            if candidate.is_dir():
                return candidate
        except PermissionError:
            ...
    return None


def _read_abbert2_model_params(run_id: int = 15) -> dict:
    with resources.files('abbert2.vendored.cerebras_bayer_shared.bert.tf') as path:
        return get_params(path / 'configs' / f'run{run_id}_params.yaml')


def _find_abbert2_checkpoints_path() -> Optional[Path]:
    """Try to infer where the checkpoints live."""

    RELATIVE_MODEL_CHECKPOINTS_PATH = Path(__file__).parent.parent.parent / 'models' / 'final_trained_models'
    try:
        from antidoto.data import ANTIDOTO_MODELS_PATH
    except ImportError:
        ANTIDOTO_MODELS_PATH = None

    candidates = (
        # Environment variable first
        os.getenv('ABBERT2_CHECKPOINTS_PATH', None),
        # Relative path to models
        RELATIVE_MODEL_CHECKPOINTS_PATH,
        # Default path in the Bayer data lake
        ANTIDOTO_MODELS_PATH / 'abbert2' / 'final_trained_models' if ANTIDOTO_MODELS_PATH else None,
    )

    candidate = _find_first_existing_directory(candidates)
    if candidate:
        return candidate

    raise FileNotFoundError(f'Could not find abbert2 checkpoints.'
                            f'\nPlease define the ABBERT2_CHECKPOINTS_PATH environment variable '
                            f'or copy / link it to {RELATIVE_MODEL_CHECKPOINTS_PATH}')


def _find_abbert2_model_checkpoint(
    run_id: int = 15,
    checkpoint_step: Optional[int] = None,
    checkpoints_path: Path = None
) -> Path:

    # cerebras uses estimator checkpoint utils for this

    if checkpoints_path is None:
        checkpoints_path = _find_abbert2_checkpoints_path()

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


# --- Wrapper to keep the TF session open and coveniently allow multiple calls with different sequence iterators

class InfiniteAbbert2(ExtractEmbeddingsFromBert):

    def __init__(self, run_id=15, remove_cls=True, remove_sep=True, warmup=True):
        super().__init__(
            params=_read_abbert2_model_params(run_id=run_id),
            checkpoint_path=str(_find_abbert2_model_checkpoint(run_id=run_id)),
            args=None
        )
        self.remove_cls = remove_cls
        self.remove_sep = remove_sep
        self._embedder_cache = None
        self._next = None
        if warmup:
            self.warmup()

    # --- Iterator with len 0 to trick clunky cerebras BERTs

    def __len__(self):
        return 0

    def __iter__(self):
        return self

    def __next__(self):
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
        #
        # For example:
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


# --- Model summaries
# From report + emails + excel file (beware: super copy-pasta)
# You can find the original sources, with extended info, in the checkpoints directory

MODELS_ARQUITECTURES = {
    'bert-small': {'L':  4, 'A':  8, 'H':  128},
    'bert-base':  {'L': 12, 'A': 12, 'H':  768},
    'bert-large': {'L': 12, 'A': 16, 'H': 1024},
}

ABBERT2_MODELS = (
    {
        'run_id': 1,
        'chain': 'heavy',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 1e-4,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 470_000,
        'total_steps@bsz1024': 500_000,
        'overall_accuracy': 0.85,
        'cdr3_accuracy': 0.57,
        'samples/s': 7_558,
        'pe_utilization': 63,
    },

    {
        'run_id': 2,
        'chain': 'heavy',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': True,
        'lr': 1e-4,
        'notes': 'segment_embedding',
        'checkpoint_best_cdr3_accuracy': 490_000,
        'total_steps@bsz1024': 500_000,
        'overall_accuracy': 0.85,
        'cdr3_accuracy': 0.57,
        'samples/s': 7_455,
        'pe_utilization': 62,
    },

    {
        'run_id': 3,
        'chain': 'both',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 1e-4,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 650_000,
        'total_steps@bsz1024': 684_809,
        'overall_accuracy': 0.83,
        'cdr3_accuracy': 0.56,
        'samples/s': 7_580,
        'pe_utilization': 63,
    },

    {
        'run_id': 4,
        'chain': 'heavy',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-large',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 1e-4,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 310_000,
        'total_steps@bsz1024': 340_000,
        'overall_accuracy': 0.85,
        'cdr3_accuracy': 0.57,
        'samples/s': 4_547,
        'pe_utilization': 94,
    },

    {
        'run_id': 5,
        'chain': 'both',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 1e-4,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 400_000,
        'total_steps@bsz1024': 2_000_000,
        'overall_accuracy': 0.85,
        'cdr3_accuracy': 0.58,
        'samples/s': 7_512,
        'pe_utilization': 63,
    },

    {
        'run_id': 6,
        'chain': 'both',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-base',
        'masking': 'cdr1=25%,cdr2=25%,cdr3=50%',
        'segment_embedding': False,
        'lr': 1e-4,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 200_000,
        'total_steps@bsz1024': 200_000,
        'overall_accuracy': 0.71,
        'cdr3_accuracy': 0.57,
        'samples/s': 7_485,
        'pe_utilization': 63,
    },

    {
        'run_id': 7,
        'chain': 'both',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-base',
        'masking': 'cdr1=25%,cdr2=25%,cdr3=50%',
        'segment_embedding': False,
        'lr': 4e-4,
        'notes': '5M shuffle buffer',
        'checkpoint_best_cdr3_accuracy': 50_000,
        'total_steps@bsz1024': 50_000,
        'overall_accuracy': 0.70,
        'cdr3_accuracy': 0.54,
        'samples/s': 7_498,
        'pe_utilization': 63,
    },

    {
        'run_id': 8,
        'chain': 'both',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-base',
        'masking': 'cdr1=25%,cdr2=25%,cdr3=50%',
        'segment_embedding': False,
        'lr': 4e-4,
        'notes': '500k shuffle buffer',
        'checkpoint_best_cdr3_accuracy': 490_000,
        'total_steps@bsz1024': 500_000,
        'overall_accuracy': 0.72,
        'cdr3_accuracy': 0.58,
        'samples/s': 7_493,
        'pe_utilization': 63,
    },

    {
        'run_id': 9,
        'chain': 'both',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-base',
        'masking': 'cdr1=25%,cdr2=25%,cdr3=50%',
        'segment_embedding': False,
        'lr': 6e-4,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 450_000,
        'total_steps@bsz1024': 540_000,
        'overall_accuracy': 0.71,
        'cdr3_accuracy': 0.58,
        'samples/s': 7_497,
        'pe_utilization': 63,
    },

    {
        'run_id': 10,
        'chain': 'both',
        'species': 'human',
        'filters': '1-2,year<2018',
        'max_sequence_length': 164,
        'model': 'bert-small',
        'masking': 'cdr1=25%,cdr2=25%,cdr3=50%',
        'segment_embedding': False,
        'lr': 4e-4,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 480_000,
        'total_steps@bsz1024': 500_000,
        'overall_accuracy': 0.65,
        'cdr3_accuracy': 0.53,
        'samples/s': 10_180,
        'pe_utilization': 2,
    },

    # --- Here we changed the dataset and filters
    #
    # Accuracy becomes better, might be real or an artifact of eval.
    #
    # We become a tad slower, likely because of larger sequence length
    # and maybe also due to larger data (I/O bound?)
    # => Hard to say, as there are no reports of CS2 utilization
    #

    {
        'run_id': 11,
        'chain': 'both',
        'species': 'all',
        'filters': '1-11,hash split',
        'max_sequence_length': 182,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 1e-4,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 300_000,
        'total_steps@bsz1024': 300_000,
        'overall_accuracy': 0.90,
        'cdr3_accuracy': 0.68,
        'samples/s': 6_861,
        'pe_utilization': 62,
    },

    {
        'run_id': 12,
        'chain': 'both',
        'species': 'all',
        'filters': '1-11,hash split',
        'max_sequence_length': 182,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 1e-6,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 300_000,
        'total_steps@bsz1024': 750_000,
        'overall_accuracy': 0.89,
        'cdr3_accuracy': 0.62,
        'samples/s': 6_909,
        'pe_utilization': 63,
    },

    {
        'run_id': 13,
        'chain': 'both',
        'species': 'all',
        'filters': '1-11,hash split',
        'max_sequence_length': 182,
        'model': 'bert-base',
        'masking': 'cdr1=12.5,cdr2=12.5,cdr3=25,fw=12.5',
        'segment_embedding': False,
        'lr': 1e-6,
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 300_000,
        'total_steps@bsz1024': 750_000,
        'overall_accuracy': 0.83,
        'cdr3_accuracy': 0.62,
        'samples/s': 6_906,
        'pe_utilization': 63,
    },

    # And here comes the big jump, use LRS

    {
        'run_id': 14,
        'chain': 'both',
        'species': 'all',
        'filters': '1-11,hash split',
        'max_sequence_length': 182,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 'lrs',
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 650_000,
        'total_steps@bsz1024': 800_000,
        'overall_accuracy': 0.92,
        'cdr3_accuracy': 0.71,
        'samples/s': 6_955,
        'pe_utilization': 62,
    },

    {
        'run_id': 15,
        'chain': 'both',
        'species': 'all',
        'filters': '1-11,hash split',
        'max_sequence_length': 182,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 'lrs',
        'notes': 'weight CDR3 loss 2x',
        'checkpoint_best_cdr3_accuracy': 300_000,
        'total_steps@bsz1024': 300_000,
        'overall_accuracy': 0.91,
        'cdr3_accuracy': 0.71,
        'samples/s': 6_830,
        'pe_utilization': 63,
    },

    # Note the performance deterioration if we focus on human

    {
        'run_id': 16,
        'chain': 'heavy',
        'species': 'human',
        'filters': '1-11,hash split',
        'max_sequence_length': 182,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 'lrs',
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 400_000,
        'total_steps@bsz1024': 447_393,
        'overall_accuracy': 0.83,
        'cdr3_accuracy': 0.53,
        'samples/s': 6_932,
        'pe_utilization': 63,
    },

    {
        'run_id': 17,
        'chain': 'both',
        'species': 'human',
        'filters': '1-11,hash split',
        'max_sequence_length': 182,
        'model': 'bert-base',
        'masking': 'uniform 15%',
        'segment_embedding': False,
        'lr': 'lrs',
        'notes': None,
        'checkpoint_best_cdr3_accuracy': 550_000,
        'total_steps@bsz1024': 551_438,
        'overall_accuracy': 0.84,
        'cdr3_accuracy': 0.57,
        'samples/s': 6_901,
        'pe_utilization': 63,
    },
)


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
