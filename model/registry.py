from .rdn import RDN
from .metaformer import MetaFormerEncoder

ENCODER_REGISTRY = {
    "rdn": RDN,
    "metaformer": MetaFormerEncoder,
}


def build_encoder(name: str, params: dict):
    try:
        cls = ENCODER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown encoder '{name}'. Options: {list(ENCODER_REGISTRY)}"
        ) from e
    return cls(**params)
