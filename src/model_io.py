# """Save/load the trained pipeline with skops (secure, no arbitrary code on load)."""
# import zipfile
# from pathlib import Path

# import skops.io as sio

# # Types skops won't load unless we trust them explicitly.
# # The pipeline contains RandomForest trees + our own feature function.
# TRUSTED_TYPES = [
#     "sklearn.tree._tree.Tree",
#     "src.features.prepare_features",
# ]


# def save_model(model, path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     sio.dump(model, path, compression=zipfile.ZIP_DEFLATED, compresslevel=6)


# def load_model(path: Path):
#     unknown = set(sio.get_untrusted_types(file=path)) - set(TRUSTED_TYPES)
#     if unknown:
#         raise ValueError(f"Refusing to load {path}: untrusted types {sorted(unknown)}")
#     return sio.load(path, trusted=TRUSTED_TYPES)