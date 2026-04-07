from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from serve import runtime_paths


def _resolve_text_encoder_source(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        local_path = runtime_paths.groundingdino_text_encoder_path()
        if local_path.exists():
            return str(local_path), True
    return text_encoder_type, False


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))
    model_source, local_only = _resolve_text_encoder_source(text_encoder_type)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    except Exception as exc:
        raise RuntimeError(
            "GroundingDINO text encoder is unavailable locally. "
            "Provide a local 'bert-base-uncased' directory under checkpoints "
            f"or set SOFAR_GROUNDINGDINO_TEXT_ENCODER. Resolved source: '{model_source}'."
        ) from exc
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    model_source, local_only = _resolve_text_encoder_source(text_encoder_type)
    if text_encoder_type == "bert-base-uncased":
        try:
            return BertModel.from_pretrained(model_source, local_files_only=local_only)
        except Exception as exc:
            raise RuntimeError(
                "GroundingDINO BERT weights are unavailable locally. "
                "Provide a local 'bert-base-uncased' directory under checkpoints "
                f"or set SOFAR_GROUNDINGDINO_TEXT_ENCODER. Resolved source: '{model_source}'."
            ) from exc
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(model_source, local_files_only=local_only)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
