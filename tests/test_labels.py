from pathlib import Path

from mavis.labels import infer_label_from_path, normalize_label


def test_normalize_known_aliases() -> None:
    assert normalize_label("Unauthorized Intervention") == "unauthorized_intervention"
    assert normalize_label("Carrying Overload with Forklift") == "carrying_overload_with_forklift"


def test_infer_label_from_directory() -> None:
    path = Path("dataset/Opened Panel Cover/video001.mp4")
    assert infer_label_from_path(path) == "opened_panel_cover"


def test_infer_numbered_dataset_directory() -> None:
    path = Path("dataset/train/2_opened_panel cover/2_tr1.mp4")
    assert infer_label_from_path(path) == "opened_panel_cover"
