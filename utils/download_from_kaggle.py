from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(
    dataset: str,
    destination: str | Path,
    file_name: str | None = None,
    unzip: bool = True,
    force: bool = False,
) -> Path:
    """
    Download a Kaggle dataset or one file from it.

    Args:
        dataset: Kaggle dataset handle, e.g. "zynicide/wine-reviews".
        destination: Local folder to save downloaded content.
        file_name: Optional exact filename within the dataset. If omitted,
                   downloads all dataset files.
        unzip: Extract the downloaded archive when downloading a full dataset.
        force: Re-download even when the destination already has the file(s).

    Returns:
        The destination folder as a Path object.
    """
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    if file_name:
        api.dataset_download_file(
            dataset=dataset,
            file_name=file_name,
            path=str(destination),
            force=force,
            quiet=False,
        )
    else:
        api.dataset_download_files(
            dataset=dataset,
            path=str(destination),
            unzip=unzip,
            force=force,
            quiet=False,
        )

    return destination