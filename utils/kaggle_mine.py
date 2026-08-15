from pathlib import Path
from typing import Literal
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(
    resource: str,
    resource_type: Literal["dataset", "competition"] = "dataset",
    destination: str | Path = "./data",
    file_name: str | None = None,
    unzip: bool = True,
    delete_zip: bool = True,
) -> Path:
    """
    Download a Kaggle dataset or competition if destination is empty.

    Hidden items whose names begin with "." are ignored when deciding
    whether destination contains data.
    """
    if resource_type not in {"dataset", "competition"}:
        raise ValueError("resource_type must be 'dataset' or 'competition'.")

    destination = Path(destination).expanduser().resolve()

    print(f"Checking destination: {destination}", flush=True)

    # Treat a folder containing only hidden files (such as .DS_Store) as empty.
    if destination.exists():
        if not destination.is_dir():
            raise NotADirectoryError(
                f"Destination exists but is not a directory: {destination}"
            )

        visible_contents = [
            item for item in destination.iterdir()
            if not item.name.startswith(".")
        ]

        if visible_contents:
            print("Download skipped: destination contains existing data.", flush=True)
            print("Existing files/folders:", flush=True)
            for item in visible_contents:
                print(f"  - {item.name}", flush=True)
            return destination

    destination.mkdir(parents=True, exist_ok=True)

    print(f"Authenticating with Kaggle...", flush=True)
    api = KaggleApi()
    api.authenticate()

    print(
        f"Starting {resource_type} download: {resource}\n"
        f"Saving to: {destination}",
        flush=True,
    )

    if resource_type == "dataset":
        if file_name is not None:
            print(f"Downloading file: {file_name}", flush=True)
            api.dataset_download_file(
                dataset=resource,
                file_name=file_name,
                path=str(destination),
                force=False,
                quiet=False,  # Enables Kaggle tqdm progress when supported.
            )
        else:
            print("Downloading all dataset files...", flush=True)
            api.dataset_download_files(
                dataset=resource,
                path=str(destination),
                unzip=False,
                force=False,
                quiet=False,  # Enables Kaggle tqdm progress when supported.
            )

    else:  # competition
        if file_name is not None:
            print(f"Downloading file: {file_name}", flush=True)
            api.competition_download_file(
                competition=resource,
                file_name=file_name,
                path=str(destination),
                force=False,
                quiet=False,
            )
        else:
            print("Downloading all competition files...", flush=True)
            api.competition_download_files(
                competition=resource,
                path=str(destination),
                force=False,
                quiet=False,
            )

    print("Download finished.", flush=True)

    if unzip:
        zip_files = list(destination.glob("*.zip"))

        if not zip_files:
            print("No ZIP files found to extract.", flush=True)

        for zip_path in zip_files:
            print(f"Extracting: {zip_path.name}", flush=True)

            with ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(destination)

            print(f"Extracted: {zip_path.name}", flush=True)

            if delete_zip:
                zip_path.unlink()
                print(f"Deleted ZIP: {zip_path.name}", flush=True)

    print(f"Ready: {destination}", flush=True)
    return destination