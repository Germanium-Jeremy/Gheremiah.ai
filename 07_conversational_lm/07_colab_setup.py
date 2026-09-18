"""Colab helpers for downloading the conversational-LM datasets and checkpoints."""

import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DOWNLOADS = {
    "cornell": "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
    # The GitHub source archive does not contain the CSV data files. If the
    # CSVs are not already under repositories/, download the dataset archive.
    "empathetic": "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz",
    "convai2": "http://parl.ai/downloads/convai2/convai2_fix_723.tgz",
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def _download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        print(f"exists: {destination}")
        return destination
    print(f"downloading: {url}")
    urllib.request.urlretrieve(url, destination)
    return destination


def _local_or_download(url, local_candidates, destination):
    """Use a user-provided archive when available, otherwise download it."""
    for candidate in local_candidates:
        candidate = Path(candidate)
        if candidate.exists() and candidate.stat().st_size:
            print(f"using local archive: {candidate}")
            return candidate
    return _download(url, destination)


def _copy_named_files(root, names, destination):
    destination.mkdir(parents=True, exist_ok=True)
    found = {}
    for path in Path(root).rglob("*"):
        if path.is_file() and path.name in names:
            target = destination / path.name
            shutil.copy2(path, target)
            found[path.name] = target
    missing = sorted(set(names) - set(found))
    if missing:
        raise FileNotFoundError(f"archive did not contain: {', '.join(missing)}")
    return found


def _find_named_files(root, names):
    """Find files by basename below a manually extracted repositories folder."""
    found = {}
    for path in Path(root).rglob("*"):
        if path.is_file() and path.name in names and path.name not in found:
            found[path.name] = path
    return found


def _copy_file(source, destination):
    source = Path(source).resolve()
    destination = Path(destination)
    if source != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _normalize_existing_repositories(repositories_dir, raw_dir):
    """Copy already-extracted datasets into the layout used by preprocessing."""
    repositories_dir = Path(repositories_dir)
    raw_dir = Path(raw_dir)

    cornell_files = _find_named_files(
        repositories_dir,
        {"movie_lines.txt", "movie_conversations.txt"},
    )
    if len(cornell_files) == 2:
        destination = raw_dir / "cornell" / "cornell movie-dialogs corpus"
        destination.mkdir(parents=True, exist_ok=True)
        for name, source in cornell_files.items():
            _copy_file(source, destination / name)

    empathetic_files = _find_named_files(
        repositories_dir,
        {"train.csv", "valid.csv", "validation.csv"},
    )
    if "train.csv" in empathetic_files:
        destination = raw_dir / "empatheticdialogues"
        destination.mkdir(parents=True, exist_ok=True)
        _copy_file(empathetic_files["train.csv"], destination / "train.csv")
        validation = empathetic_files.get("valid.csv") or empathetic_files.get("validation.csv")
        if validation:
            _copy_file(validation, destination / "valid.csv")

    convai_files = _find_named_files(
        repositories_dir,
        {"train_self_original_no_cands.txt", "valid_self_original_no_cands.txt"},
    )
    for name, source in convai_files.items():
        _copy_file(source, raw_dir / name)


def download_raw_datasets(project_root, include_shakespeare=True):
    """Download archives and create the layout expected by ``07_preprocess.py``."""
    project_root = Path(project_root).resolve()
    repositories_dir = project_root / "repositories"
    raw_dir = project_root / "repositories" / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _normalize_existing_repositories(repositories_dir, raw_dir)

    required = [
        raw_dir / "cornell" / "cornell movie-dialogs corpus" / "movie_lines.txt",
        raw_dir / "cornell" / "cornell movie-dialogs corpus" / "movie_conversations.txt",
        raw_dir / "empatheticdialogues" / "train.csv",
        raw_dir / "empatheticdialogues" / "valid.csv",
        raw_dir / "train_self_original_no_cands.txt",
        raw_dir / "valid_self_original_no_cands.txt",
    ]
    shakespeare_path = project_root / "dataset" / "input.txt"
    if all(path.exists() for path in required) and (
        not include_shakespeare or shakespeare_path.exists()
    ):
        print(f"raw datasets already ready: {raw_dir}")
        return raw_dir

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)

        cornell_zip = _local_or_download(
            DOWNLOADS["cornell"],
            (repositories_dir / "cornell.zip", raw_dir / "cornell.zip"),
            temporary / "cornell.zip",
        )
        with zipfile.ZipFile(cornell_zip) as archive:
            archive.extractall(temporary / "cornell_extract")
        _copy_named_files(
            temporary / "cornell_extract",
            {"movie_lines.txt", "movie_conversations.txt"},
            raw_dir / "cornell" / "cornell movie-dialogs corpus",
        )

        empathetic_archive = _local_or_download(
            DOWNLOADS["empathetic"],
            (
                repositories_dir / "empatheticdialogues.tar.gz",
                raw_dir / "empatheticdialogues.tar.gz",
            ),
            temporary / "empatheticdialogues.tar.gz",
        )
        with tarfile.open(empathetic_archive, mode="r:gz") as archive:
            archive.extractall(temporary / "empathetic_extract")
        _copy_named_files(
            temporary / "empathetic_extract",
            {"train.csv", "valid.csv"},
            raw_dir / "empatheticdialogues",
        )

        convai_archive = _local_or_download(
            DOWNLOADS["convai2"],
            (repositories_dir / "convai2_fix_723.tgz", raw_dir / "convai2_fix_723.tgz"),
            temporary / "convai2.tgz",
        )
        with tarfile.open(convai_archive, mode="r:gz") as archive:
            archive.extractall(temporary / "convai2_extract")
        _copy_named_files(
            temporary / "convai2_extract",
            {"train_self_original_no_cands.txt", "valid_self_original_no_cands.txt"},
            raw_dir,
        )

    if include_shakespeare:
        _download(DOWNLOADS["shakespeare"], shakespeare_path)

    print(f"raw datasets ready: {raw_dir}")
    return raw_dir


def mount_google_drive(mount_point="/content/drive"):
    """Mount Google Drive when this code is running inside Colab."""
    from google.colab import drive

    drive.mount(mount_point)
    return Path(mount_point) / "MyDrive"


def checkpoint_dir(project_root, use_google_drive=False, drive_folder="Gheremiah"):
    """Return a persistent model directory for local Colab or Google Drive storage."""
    project_root = Path(project_root).resolve()
    if use_google_drive:
        model_dir = mount_google_drive() / drive_folder / "models"
    else:
        model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint directory: {model_dir}")
    return model_dir