import yaml
import collections
import tensorflow as tf 
from modelzoo.transformers.bert.tf.utils import set_defaults as set_bert_defaults
from copy import deepcopy

def get_oas_vocab(vocab_file=None, dummy_vocab_size=None):
    """
    Returns the vocabulary used by the oas pipe as a dictionary
    along with the min/max amino acid ids, such that a random amino
    acid can be generated using:
    
        random.randint(min_aa_id, max_aa_id)

    Letter codes standard for IUPAC. 
    See here: https://www.bioinformatics.org/sms2/iupac.html
    """
    
    if vocab_file:
        vocab = collections.OrderedDict()
        with open(vocab_file, "r") as fh:
            data = fh.readlines()

        for i, key in enumerate(data):
            key = key.strip()
            vocab[key] = i
    
    else:
        vocab = {
            "[PAD]": 0,
            "[MASK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[UNK]": 4,
            "A": 5,
            "B": 6,
            "C": 7,
            "D": 8,
            "E": 9,
            "F": 10,
            "G": 11,
            "H": 12,
            "I": 13,
            "J": 14,
            "K": 15,
            "L": 16,
            "M": 17,
            "N": 18,
            "O": 19,
            "P": 20,
            "Q": 21,
            "R": 22,
            "S": 23,
            "T": 24,
            "U": 25,
            "V": 26,
            "W": 27,
            "X": 28,
            "Y": 29,
            "Z": 30,
        }
    min_aa_id = vocab["A"]
    max_aa_id = vocab["Z"]

    len_vocab = len(vocab)
    if dummy_vocab_size and dummy_vocab_size > len(vocab):
        extra_ids = dummy_vocab_size - len(vocab)
        for i in range(extra_ids):
            extra_key = "extra_id_"+ str(i)
            vocab[extra_key] = i + len_vocab
    
    tf.compat.v1.logging.info(f"----vocab: {vocab}, len(vocab): {len(vocab)}")
    return vocab, min_aa_id, max_aa_id


def get_params(params_file, mode=None):

    # Load yaml into params.
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)

    vocab_size = len(get_oas_vocab(params["train_input"].get("vocab_file"), params["train_input"].get("dummy_vocab_size"))[0])

    params["train_input"]["vocab_size"] = vocab_size

    if "eval_input" in params:
        params["eval_input"]["vocab_size"] = vocab_size
    

    params["model"]["all_encoder_outputs"] = params["model"].get("all_encoder_outputs", False)

    params["train_input"]["fw_masked_lm_prob"] = params["train_input"].get("fw_masked_lm_prob", None)
    params["train_input"]["cdr_masked_lm_prob"] = params["train_input"].get("cdr_masked_lm_prob", None)

    params["eval_input"]["fw_masked_lm_prob"] = params["train_input"].get("fw_masked_lm_prob", None)
    params["eval_input"]["cdr_masked_lm_prob"] = params["train_input"].get("cdr_masked_lm_prob", None)

    if "predict_input" in params:
        vocab, min_aa_id, max_aa_id = get_oas_vocab(params["predict_input"].get("vocab_file"), params["predict_input"].get("dummy_vocab_size"))
        params["predict_input"]["vocab_size"] = len(vocab)
    else:
        params["predict_input"] = deepcopy(params["train_input"])


    # predict_input required parameters
    params["predict_input"]["shuffle"] = params["predict_input"].get("shuffle", False)
    params["predict_input"]["repeat"] = params["predict_input"].get("repeat", False)

    params["predict_input"]["mixed_precision"] = params["model"].get(
        "mixed_precision", False
    )

    set_bert_defaults(params, mode=mode)

    return params


