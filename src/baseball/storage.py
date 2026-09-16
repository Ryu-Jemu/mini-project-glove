"""오브젝트 스토리지(MinIO) 계층. 키 규약은 콘텐츠 주소 rules/<sha256>.pdf."""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from minio import Minio
from minio.error import S3Error

from baseball.config import Settings, get_settings

# ingest 대상 allowlist (glob 금지 — Lorem_ipsum.pdf 등 잡음 배제)
RULEBOOK_FILENAME = "2026_야구규칙.pdf"
ALLOWLIST = (RULEBOOK_FILENAME,)
KEY_PREFIX = "rules/"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def object_key(digest: str) -> str:
    return f"{KEY_PREFIX}{digest}.pdf"


class ObjectStorage(Protocol):
    def ensure_bucket(self) -> bool: ...
    def upload_if_changed(self, path: Path) -> tuple[str, str]: ...
    def fetch_to_tmp(self, key: str) -> Path: ...
    def presign(self, key: str, minutes: int = 15) -> str: ...
    def list_keys(self, prefix: str = KEY_PREFIX) -> list[str]: ...


class MinioStorage:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.bucket = self.settings.minio_bucket
        self.client = Minio(
            self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key.get_secret_value(),
            secure=False,
        )

    def ensure_bucket(self) -> bool:
        """버킷이 없으면 생성. 새로 만들었으면 True."""
        if self.client.bucket_exists(self.bucket):
            return False
        self.client.make_bucket(self.bucket)
        return True

    def upload_if_changed(self, path: Path) -> tuple[str, str]:
        """콘텐츠 주소 키이므로 stat_object 존재 = 내용 동일. (key, 'uploaded'|'unchanged')"""
        digest = sha256_file(path)
        key = object_key(digest)
        self.ensure_bucket()
        try:
            self.client.stat_object(self.bucket, key)
            return key, "unchanged"
        except S3Error as exc:
            if exc.code not in {"NoSuchKey", "NoSuchObject", "NotFound"}:
                raise
        self.client.fput_object(
            self.bucket, key, str(path),
            content_type="application/pdf",
            # 사용자 메타데이터는 US-ASCII만 허용 → 한글 파일명은 percent-encoding
            metadata={"sha256": digest, "filename": quote(path.name)},
        )
        return key, "uploaded"

    def fetch_to_tmp(self, key: str) -> Path:
        tmp = Path(tempfile.gettempdir()) / f"baseball-{key.replace('/', '_')}"
        if not tmp.exists() or tmp.stat().st_size == 0:
            self.client.fget_object(self.bucket, key, str(tmp))
        return tmp

    def presign(self, key: str, minutes: int = 15) -> str:
        return self.client.presigned_get_object(
            self.bucket, key, expires=timedelta(minutes=minutes)
        )

    def list_keys(self, prefix: str = KEY_PREFIX) -> list[str]:
        return [o.object_name for o in self.client.list_objects(self.bucket, prefix=prefix, recursive=True)]

    def remove(self, key: str) -> None:
        self.client.remove_object(self.bucket, key)


def allowlisted_paths(settings: Settings | None = None) -> list[Path]:
    settings = settings or get_settings()
    return [settings.base_dir / "data" / name for name in ALLOWLIST]


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m baseball.storage")
    ap.add_argument("command", choices=["sync", "url", "list"])
    ap.add_argument("--prune", action="store_true", help="allowlist 밖 객체 나열(삭제는 --yes 필요)")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    settings = get_settings()
    store = MinioStorage(settings)
    created = store.ensure_bucket()
    if created:
        print(f"bucket created: {settings.minio_bucket}")

    keep: set[str] = set()
    for path in allowlisted_paths(settings):
        if not path.exists():
            print(f"missing: {path}")
            return 1
        key, status = store.upload_if_changed(path)
        keep.add(key)
        if args.command == "sync":
            print(f"{status} key={key} ({path.name})")
        elif args.command == "url":
            print(store.presign(key))

    if args.command == "list":
        for k in store.list_keys():
            print(k)
    if args.prune:
        orphans = [k for k in store.list_keys() if k not in keep]
        for k in orphans:
            print(f"orphan: {k}" + (" -> removed" if args.yes else ""))
            if args.yes:
                store.remove(k)
        print(f"prune candidates: {len(orphans)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
