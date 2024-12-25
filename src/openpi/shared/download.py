import concurrent.futures
import getpass
import logging
import os
import pathlib
import shutil
import stat
import time
import urllib.parse

import boto3
import boto3.s3.transfer as s3_transfer
import botocore
import filelock
import fsspec
import fsspec.generic
import s3transfer.futures as s3_transfer_futures
import tqdm_loggable.auto as tqdm
from types_boto3_s3.service_resource import ObjectSummary

# Environment variable to control cache directory path, ~/.cache/openpi will be used by default.
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"

logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path:
    default_dir = "~/.cache/openpi"
    if os.path.exists("/mnt/weka"):  # noqa: PTH110
        default_dir = f"/mnt/weka/{getpass.getuser()}/.cache/openpi"

    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, default_dir)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(url: str, **kwargs) -> pathlib.Path:
    """Download a file or directoryfrom a remote filesystem to the local cache, and return the local path.

    If the local file already exists, it will be returned directly.

    It is safe to call this function concurrently from multiple processes.
    See `get_cache_dir` for more details on the cache directory.

    Args:
        url: URL to the file to download.
        **kwargs: Additional arguments to pass to fsspec.

    Returns:
        Local path to the downloaded file or directory. That path is guaranteed to exist and is absolute.
    """
    # Don't use fsspec to parse the url to avoid unnecessary connection to the remote filesystem.
    parsed = urllib.parse.urlparse(url)

    # Short circuit if this is a local path.
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    local_path = get_cache_dir() / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()

    # Check if file already exists in cache.
    if local_path.exists():
        return local_path

    # Download file from remote file system.
    logger.info(f"Downloading {url} to {local_path}")
    with filelock.FileLock(local_path.with_suffix(".lock")):
        scratch_path = local_path.with_suffix(".partial")

        if _is_openpi_url(url):
            # Download with openpi credentials.
            # TODO(ury): Remove once the bucket becomes public.
            boto_session = boto3.session.Session(
                aws_access_key_id="AKIA4MTWIIQIZBO44C62",
                aws_secret_access_key="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",
                region_name="us-east-1",
            )
            _download_boto3(url, scratch_path, boto_session=boto_session)
        elif url.startswith("s3://"):
            # Download with default boto3 credentials.
            _download_boto3(url, scratch_path)
        else:
            _download_fsspec(url, scratch_path, **kwargs)

        shutil.move(scratch_path, local_path)
        _ensure_permissions(local_path)

    return local_path


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    """Download a file from a remote filesystem to the local cache, and return the local path."""
    fs, _ = fsspec.core.url_to_fs(url, **kwargs)
    info = fs.info(url)
    if is_dir := (info["type"] == "directory"):  # noqa: SIM108
        total_size = fs.du(url)
    else:
        total_size = info["size"]
    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, local_path, recursive=is_dir)
        while not future.done():
            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        pbar.update(total_size - pbar.n)


def _download_boto3(
    url: str,
    local_path: pathlib.Path,
    *,
    boto_session: boto3.Session | None = None,
    workers: int = 20,
) -> None:
    """Download a file from the OpenPI S3 bucket using boto3. This is a more performant version of download but can
    only handle s3 urls. In openpi repo, this is mainly used to access assets in S3 with higher throughput.

    Input:
        url: URL to openpi checkpoint path.
        local_path: local path to the downloaded file.
        boto_session: Optional boto3 session, will create by default if not provided.
        workers: number of workers for downloading.
    """

    def validate_and_parse_url(maybe_s3_url: str) -> tuple[str, str]:
        parsed = urllib.parse.urlparse(maybe_s3_url)
        if parsed.scheme != "s3":
            raise ValueError(f"URL must be an S3 URL (s3://), got: {maybe_s3_url}")
        bucket_name = parsed.netloc
        prefix = parsed.path.strip("/")
        return bucket_name, prefix

    bucket_name, prefix = validate_and_parse_url(url)
    session = boto_session or boto3.Session()

    s3api = session.resource("s3")
    bucket = s3api.Bucket(bucket_name)

    objects = list(bucket.objects.filter(Prefix=prefix))
    total_size = sum(obj.size for obj in objects)

    s3t = _get_s3_transfer_manager(session, workers)

    def transfer(
        s3obj: ObjectSummary, dest_path: pathlib.Path, progress_func
    ) -> s3_transfer_futures.TransferFuture | None:
        if dest_path.exists():
            dest_stat = dest_path.stat()
            if s3obj.size == dest_stat.st_size:
                progress_func(s3obj.size)
                return None
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        return s3t.download(
            bucket_name,
            s3obj.key,
            str(dest_path),
            subscribers=[
                s3_transfer.ProgressCallbackInvoker(progress_func),
            ],
        )

    try:
        with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
            futures = []
            for obj in objects:
                relative_path = pathlib.Path(obj.key).relative_to(prefix)
                dest_path = local_path / relative_path
                if future := transfer(obj, dest_path, pbar.update):
                    futures.append(future)
            for future in futures:
                future.result()
    finally:
        s3t.shutdown()


def _get_s3_transfer_manager(session: boto3.Session, workers: int) -> s3_transfer.TransferManager:
    botocore_config = botocore.config.Config(max_pool_connections=workers)
    s3client = session.client("s3", config=botocore_config)
    transfer_config = s3_transfer.TransferConfig(
        use_threads=True,
        max_concurrency=workers,
    )
    return s3_transfer.create_transfer_manager(s3client, transfer_config)


def _set_permission(path: pathlib.Path, target_permission: int):
    """chmod requires executable permission to be set, so we skip if the permission is already match with the target."""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """Set folder permission to be read, write and searchable."""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """Since we are sharing cache directory with containerized runtime as well as training script, we need to
    ensure that the cache directory has the correct permissions.
    """

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """Set all files to be read & writable, if it is a script, keep it as a script."""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _is_openpi_url(url: str) -> bool:
    """Check if the url is an OpenPI S3 bucket url."""
    return url.startswith("s3://openpi-assets/")
