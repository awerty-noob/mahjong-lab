"""Trusted checkpoint envelopes, contract digests, and retention."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import pickle
import re
import shutil
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from .continuation import _checked_array, _copy_metadata, _counter, _equal_contract, _mapping, _rng
from networks.mahjong_3p.configs import (
    DEFAULT_CRITIC_CONFIG,
    RULESET,
    validate_critic_config,
    validate_head_config,
    validate_network_config,
    validate_reading_config,
)
from train.configs import VRPOTrainingState, validate_reward_config
from train.selfplay import checked_collection_state

# --- from checkpoint_store.py ---


DEFAULT_MIN_FREE_BYTES = 4 * 1024**3
_SCHEMA = 1
_OWNER = "owner.json"
_LEDGER = "ownership.jsonl"
_LATEST = "latest.json"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class CheckpointStoreError(RuntimeError):
    """Ownership, integrity, or publication contract violation."""


class InsufficientSpaceError(CheckpointStoreError):
    """The complete temporary pair plus metadata would invade the reserve."""


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class CheckpointStore:
    """Create a NEW owned run; explicit classmethod handles restart.

    ``config_source`` is the complete configuration source (UTF-8 text/bytes),
    retained verbatim in the immutable owner marker with its byte SHA256. Each
    pair points to that source. Pass resolved configuration, not just a pathname.
    ``archive_interval=0`` disables model milestones. Resume payloads are never
    archived. Archive models retain their original ordinary-model filename.
    """

    def __init__(
        self,
        directory,
        run_id,
        *,
        keep_last=2,
        archive_interval=25,
        min_free_bytes=DEFAULT_MIN_FREE_BYTES,
        config_source="",
    ):
        self._init_path(directory, run_id)
        _integer(keep_last, "keep_last", 2)
        _integer(archive_interval, "archive_interval")
        _integer(min_free_bytes, "min_free_bytes")
        if isinstance(config_source, str):
            source = config_source.encode("utf-8")
        elif isinstance(config_source, bytes):
            source = config_source
        else:
            raise TypeError("config_source must be UTF-8 str or bytes")
        text = source.decode("utf-8")
        # mkdir is exclusive: never adopt or clean an existing directory.
        os.mkdir(self.directory, 0o700)
        self._open_root()
        self.owner_id = uuid.uuid4().hex
        self._owner = {
            "schema": _SCHEMA,
            "run_id": run_id,
            "owner_id": self.owner_id,
            "directory": str(self.directory),
            "device": self._identity[0],
            "inode": self._identity[1],
            "keep_last": keep_last,
            "archive_interval": archive_interval,
            "min_free_bytes": min_free_bytes,
            "config_source": {"text": text, "bytes": len(source), "sha256": _sha(source), "encoding": "utf-8"},
        }
        self._owner_bytes = _json(self._owner)
        try:
            self._exclusive_write(_OWNER, self._owner_bytes)
            self._open_ledger(create=True)
            self._fsync_directory()
            # Persist the newly created run directory entry as well.
            fd = os.open(self.directory.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            self._load_journal()
            self._append_event({"type": "created", "owner_sha256": _sha(self._owner_bytes)})
        except BaseException:
            self.close()
            raise

    @classmethod
    def create(cls, directory, run_id, **kwargs):
        return cls(directory, run_id, **kwargs)

    @classmethod
    def open_existing(cls, directory, run_id, *, owner_id):
        """Reopen only one's own run, with matching immutable policy and ledger.

        Unknown files, modified published files, symlinks, copied/renamed
        directories, and a wrong owner token are refused. Incomplete reserved
        orphans are not adopted or deleted; later publications leave them for
        audit. No repair/deletion occurs on open.
        """
        if not isinstance(owner_id, str) or not _TOKEN.fullmatch(owner_id):
            raise CheckpointStoreError("an explicit valid owner_id is required")
        self = cls.__new__(cls)
        self._init_path(directory, run_id)
        self._open_root()
        try:
            raw = self._read_regular(_OWNER)
            owner = json.loads(raw)
            if (
                owner.get("schema") != _SCHEMA
                or owner.get("run_id") != run_id
                or owner.get("owner_id") != owner_id
                or owner.get("directory") != str(self.directory)
                or (owner.get("device"), owner.get("inode")) != self._identity
                or raw != _json(owner)
            ):
                raise CheckpointStoreError("immutable owner marker mismatch")
            _integer(owner["keep_last"], "keep_last", 2)
            _integer(owner["archive_interval"], "archive_interval")
            _integer(owner["min_free_bytes"], "min_free_bytes")
            source = owner["config_source"]
            encoded = source["text"].encode("utf-8")
            if source["encoding"] != "utf-8" or len(encoded) != source["bytes"] or _sha(encoded) != source["sha256"]:
                raise CheckpointStoreError("configuration source mismatch")
            self._owner, self._owner_bytes = owner, raw
            self.owner_id = owner_id
            self._open_ledger(create=False)
            self._load_journal()
            if not self._events or self._events[0]["type"] != "created":
                raise CheckpointStoreError("ownership ledger lacks durable creation binding")
            self._check_inventory(verify=True)
            self._latest_record(verify=True)
            return self
        except BaseException:
            self.close()
            raise

    def _init_path(self, directory, run_id):
        if not isinstance(run_id, str) or not _RUN.fullmatch(run_id):
            raise ValueError("run_id must be a simple nonempty identifier")
        path = Path(directory).expanduser()
        if ".." in path.parts:
            raise CheckpointStoreError("directory traversal is forbidden")
        self.directory = path.absolute()
        if self.directory.parent.resolve(strict=True) != self.directory.parent:
            raise CheckpointStoreError("symlinked parent directories are forbidden")
        self.run_id = run_id
        self._dirfd = self._journal_fd = None
        self._write_blocked = False
        self._created = {}

    def _open_root(self):
        self._dirfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(self._dirfd)
        self._identity = (info.st_dev, info.st_ino)
        self._check_root()

    def _check_root(self):
        if self._dirfd is None:
            raise CheckpointStoreError("store is closed")
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or (info.st_dev, info.st_ino) != self._identity
            or self.directory.resolve(strict=True) != self.directory
        ):
            raise CheckpointStoreError("owned root was replaced, moved, or symlinked")

    def _path(self, name):
        self._check_root()
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise CheckpointStoreError("invalid generated filename")
        target = self.directory / name
        # Check the intended absolute path before any unlink/replace operation.
        if target.parent != self.directory or target.resolve(strict=False) != target:
            raise CheckpointStoreError("target escapes owned root or is a symlink")
        return target

    def _regular_owned_link(self, name, info):
        if not stat.S_ISREG(info.st_mode):
            return False
        if info.st_nlink == 1:
            return True
        # A power loss between link() and unlink() may leave precisely the
        # reserved temp/final pair. Never accept any external hardlink.
        owned = getattr(self, "_owned", {})
        if info.st_nlink != 2 or name not in owned:
            return False
        peer = name[1:-4] if name.startswith(".") and name.endswith(".tmp") else "." + name + ".tmp"
        if peer not in owned or owned[peer] != owned[name]:
            return False
        self._path(peer)
        try:
            other = os.stat(peer, dir_fd=self._dirfd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return (
            stat.S_ISREG(other.st_mode)
            and other.st_nlink == 2
            and (other.st_dev, other.st_ino) == (info.st_dev, info.st_ino)
        )

    def _read_regular(self, name):
        self._path(name)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self._dirfd)
        try:
            info = os.fstat(fd)
            if not self._regular_owned_link(name, info):
                raise CheckpointStoreError(f"not an exclusive regular file: {name}")
            chunks = []
            while chunk := os.read(fd, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _exclusive_write(self, name, data):
        self._path(name)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._dirfd)
        info = os.fstat(fd)
        self._created[name] = (info.st_dev, info.st_ino)
        try:
            view = memoryview(data)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("short checkpoint write")
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _fsync_directory(self):
        self._check_root()
        os.fsync(self._dirfd)

    def _open_ledger(self, *, create):
        self._path(_LEDGER)
        flags = os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        self._journal_fd = os.open(_LEDGER, flags, 0o600, dir_fd=self._dirfd)
        info = os.fstat(self._journal_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CheckpointStoreError("invalid ownership ledger")
        fcntl.flock(self._journal_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._journal_identity = (info.st_dev, info.st_ino)
        if create:
            os.fsync(self._journal_fd)

    def _validate_record(self, record):
        tx = record["transaction"]
        update = _integer(record["update"], "update")
        _integer(record["envsteps"], "envsteps")
        if (
            not _TOKEN.fullmatch(tx)
            or record["schema"] != _SCHEMA
            or record["run_id"] != self.run_id
            or record["owner_id"] != self.owner_id
        ):
            raise CheckpointStoreError("foreign or invalid checkpoint record")
        interval = self._owner["archive_interval"]
        if record["archived_model"] != bool(interval and update % interval == 0):
            raise CheckpointStoreError("archive policy mismatch")
        if record["config_source"] != self._config_pointer():
            raise CheckpointStoreError("record configuration pointer mismatch")
        for kind in ("model", "resume"):
            entry = record[kind]
            expected = f"{kind}-u{update:012d}-{tx}.pkl"
            if entry["name"] != expected or not _HEX.fullmatch(entry["sha256"]):
                raise CheckpointStoreError("invalid owned payload name/hash")
            _integer(entry["size_bytes"], "size_bytes")
        if record["latest_temp"] != f".latest-{tx}.tmp":
            raise CheckpointStoreError("invalid manifest temporary name")

    def _config_pointer(self):
        source = self._owner["config_source"]
        return {
            "owner_marker": _OWNER,
            "sha256": source["sha256"],
            "size_bytes": source["bytes"],
            "field": "config_source.text",
        }

    def _reservation_files(self, record):
        result = {}
        for kind in ("model", "resume"):
            item = record[kind]
            value = (item["sha256"], item["size_bytes"])
            result[item["name"]] = value
            result["." + item["name"] + ".tmp"] = value
        raw = _json(record)
        result[record["latest_temp"]] = (_sha(raw), len(raw))
        return result

    def _load_journal(self):
        self._records, self._published, self._owned = {}, set(), {}
        self._file_transactions = {}
        self._retired = set()
        self._events = []
        self._chain = "0" * 64
        raw = self._read_regular(_LEDGER)
        complete = raw.rsplit(b"\n", 1)[0] + b"\n" if b"\n" in raw else b""
        self._write_blocked = bool(raw and not raw.endswith(b"\n"))
        for line in complete.splitlines():
            wrapper = json.loads(line)
            digest = wrapper.pop("sha256")
            if (
                digest != _sha(_json(wrapper))
                or wrapper["previous"] != self._chain
                or wrapper["sequence"] != len(self._events)
                or wrapper["owner_id"] != self.owner_id
            ):
                raise CheckpointStoreError("ownership ledger hash chain mismatch")
            self._apply_event(wrapper["event"])
            self._chain = digest
        self._journal_size = len(raw)
        self._journal_sha = _sha(raw)

    def _apply_event(self, event):
        kind = event["type"]
        if kind == "created":
            if self._events or event["owner_sha256"] != _sha(self._owner_bytes):
                raise CheckpointStoreError("immutable owner marker ledger binding mismatch")
        elif kind == "reserve":
            if not self._events or self._events[0]["type"] != "created":
                raise CheckpointStoreError("reservation precedes owned run creation")
            record = event["record"]
            self._validate_record(record)
            tx = record["transaction"]
            if tx in self._records or any(r["update"] >= record["update"] for r in self._records.values()):
                raise CheckpointStoreError("non-monotonic or duplicate reservation")
            files = self._reservation_files(record)
            if self._owned.keys() & files.keys():
                raise CheckpointStoreError("duplicate ownership reservation")
            self._records[tx] = record
            self._owned.update(files)
            self._file_transactions.update({name: tx for name in files})
        elif kind == "published":
            tx = event["transaction"]
            if tx not in self._records:
                raise CheckpointStoreError("publication without ownership reservation")
            self._published.add(tx)
        elif kind == "gc":
            if any(name not in self._owned for name in event["names"]):
                raise CheckpointStoreError("GC without exact ownership reservation")
            self._retired.update(event["names"])
        else:
            raise CheckpointStoreError("unknown ownership event")
        self._events.append(event)

    def _append_event(self, event):
        if self._write_blocked:
            raise CheckpointStoreError("torn/failed journal append: verified reads only")
        self._check_controls()
        wrapper = {"previous": self._chain, "sequence": len(self._events), "owner_id": self.owner_id, "event": event}
        digest = _sha(_json(wrapper))
        wrapper["sha256"] = digest
        data = _json(wrapper)
        try:
            view = memoryview(data)
            while view:
                count = os.write(self._journal_fd, view)
                if count <= 0:
                    raise OSError("short ledger write")
                view = view[count:]
            os.fsync(self._journal_fd)
        except BaseException:
            # An append/fsync error must not make previous latest unreadable.
            # Re-read our exact appended prefix (possibly a torn last line),
            # but never permit another write after uncertain durability.
            try:
                self._load_journal()
            finally:
                self._write_blocked = True
            raise
        self._apply_event(event)
        self._chain = digest
        raw = self._read_regular(_LEDGER)
        self._journal_size, self._journal_sha = len(raw), _sha(raw)

    def _check_controls(self):
        self._check_root()
        if self._read_regular(_OWNER) != self._owner_bytes:
            raise CheckpointStoreError("immutable owner marker was modified")
        info = os.stat(_LEDGER, dir_fd=self._dirfd, follow_symlinks=False)
        if (info.st_dev, info.st_ino) != self._journal_identity:
            raise CheckpointStoreError("ownership ledger was replaced")
        raw = self._read_regular(_LEDGER)
        if len(raw) != self._journal_size or _sha(raw) != self._journal_sha:
            raise CheckpointStoreError("ownership ledger was modified")

    def _verify_file(self, name, expected):
        data = self._read_regular(name)
        if len(data) != expected[1] or _sha(data) != expected[0]:
            raise CheckpointStoreError(f"owned file was modified: {name}")
        return data

    def _check_inventory(self, *, verify):
        self._check_controls()
        names = set(os.listdir(self._dirfd))
        unknown = names - self._owned.keys() - {_OWNER, _LEDGER, _LATEST}
        if unknown:
            raise CheckpointStoreError(f"unknown files in owned directory: {sorted(unknown)}")
        required = {self._records[tx][kind]["name"] for tx in self._published for kind in ("model", "resume")}
        missing = required - self._retired - names
        if missing:
            raise CheckpointStoreError(f"published owned files are missing: {sorted(missing)}")
        for name in names:
            self._path(name)
            info = os.stat(name, dir_fd=self._dirfd, follow_symlinks=False)
            if not self._regular_owned_link(name, info):
                raise CheckpointStoreError(f"symlink/nonexclusive file in owned directory: {name}")
            if verify and name in self._owned:
                # A reserved, never-published file may be a partial crash
                # orphan. It is not adopted for reads/GC and remains untouched.
                # Latest (including a post-manifest/pre-journal crash) is
                # separately verified unconditionally by _latest_record().
                if self._file_transactions[name] in self._published:
                    self._verify_file(name, self._owned[name])
        return names

    def _latest_record(self, *, verify):
        try:
            raw = self._read_regular(_LATEST)
        except FileNotFoundError:
            if self._published:
                raise CheckpointStoreError("published latest manifest is missing")
            return None
        record = json.loads(raw)
        tx = record.get("transaction")
        if tx not in self._records or record != self._records[tx] or raw != _json(record):
            raise CheckpointStoreError("latest manifest is not an exact owned record")
        if any(self._records[t]["update"] > record["update"] for t in self._published):
            raise CheckpointStoreError("latest manifest was rolled back")
        if verify:
            for kind in ("model", "resume"):
                item = record[kind]
                self._verify_file(item["name"], (item["sha256"], item["size_bytes"]))
        return record

    def _unlink_verified(self, name):
        # No arbitrary path, glob, or directory-recursive deletion is supported.
        if name not in self._owned:
            raise CheckpointStoreError("refuse deletion outside exact ownership ledger")
        target = self._path(name)
        if target.parent != self.directory or target.resolve(strict=False) != target:
            raise CheckpointStoreError("refuse deletion outside resolved owned root")
        self._check_controls()
        self._verify_file(name, self._owned[name])
        os.unlink(name, dir_fd=self._dirfd)

    def _install_payload(self, name, data):
        temp = "." + name + ".tmp"
        self._exclusive_write(temp, data)
        self._verify_file(temp, self._owned[temp])
        self._path(name)
        # link() atomically publishes without overwriting any preexisting name.
        os.link(temp, name, src_dir_fd=self._dirfd, dst_dir_fd=self._dirfd, follow_symlinks=False)
        # A hardlink pair is expected only inside this operation. Break the
        # temporary link after verifying both names reference the same inode.
        a = os.stat(temp, dir_fd=self._dirfd, follow_symlinks=False)
        b = os.stat(name, dir_fd=self._dirfd, follow_symlinks=False)
        if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino) or a.st_nlink != 2 or not stat.S_ISREG(a.st_mode):
            raise CheckpointStoreError("unexpected hardlink during publication")
        self._unlink_verified(temp)
        self._fsync_directory()

    def _public_record(self, record):
        result = copy.deepcopy(record)
        for kind in ("model", "resume"):
            result[kind]["path"] = str(self.directory / result[kind]["name"])
        result["manifest_path"] = str(self.directory / _LATEST)
        result["config_source"]["path"] = str(self.directory / _OWNER)
        return result

    def publish(self, model_payload, resume_payload, update, envsteps):
        """Durably publish a pair, then GC only verified owned older payloads.

        Serialization is in memory; disk admission occurs before *any* write.
        Exact serialized temporary pair bytes, manifest and conservative journal
        overhead are budgeted without crediting future deletions. A failed
        precommit leaves previous latest/pairs intact. A postcommit GC error is
        returned in ``record['gc']['errors']``, not disguised as failed publish.
        Failed updates are reserved permanently; retry with a higher update.
        """
        _integer(update, "update")
        _integer(envsteps, "envsteps")
        # Inventory checks names/inodes without rereading every growing model
        # archive on every update. Latest and every deletion are hash-checked.
        self._check_inventory(verify=False)
        previous = self._latest_record(verify=True)
        if self._write_blocked:
            raise CheckpointStoreError("torn journal tail: verified reads only")
        if self._records and update <= max(r["update"] for r in self._records.values()):
            raise CheckpointStoreError("update must exceed every reserved update")
        if previous and envsteps < previous["envsteps"]:
            raise CheckpointStoreError("envsteps must not decrease")
        model = pickle.dumps(model_payload, protocol=pickle.HIGHEST_PROTOCOL)
        resume = pickle.dumps(resume_payload, protocol=pickle.HIGHEST_PROTOCOL)
        tx = uuid.uuid4().hex
        interval = self._owner["archive_interval"]
        record = {
            "schema": _SCHEMA,
            "run_id": self.run_id,
            "owner_id": self.owner_id,
            "transaction": tx,
            "update": update,
            "envsteps": envsteps,
            "archived_model": bool(interval and update % interval == 0),
            "config_source": self._config_pointer(),
            "latest_temp": f".latest-{tx}.tmp",
        }
        for kind, data in (("model", model), ("resume", resume)):
            record[kind] = {"name": f"{kind}-u{update:012d}-{tx}.pkl", "sha256": _sha(data), "size_bytes": len(data)}
        manifest = _json(record)
        reserve = {"type": "reserve", "record": record}
        # Two full payloads are temporarily present in addition to ALL old
        # pairs. link/rename do not copy their bytes. Metadata margin includes
        # reservation, publication, recovery and all pending GC journal events.
        estimated = (
            len(model)
            + len(resume)
            + len(manifest)
            + 4 * len(_json(reserve))
            + 16384
            + sum(len(_json(e)) + 1024 for e in self._events if e["type"] == "reserve")
        )
        free = shutil.disk_usage(self.directory).free
        if free < self._owner["min_free_bytes"] + estimated:
            raise InsufficientSpaceError(
                f"free={free}, estimated_temporary_bytes={estimated}, reserve={self._owner['min_free_bytes']}"
            )
        if previous and previous["transaction"] not in self._published:
            # Recover a crash after latest fsync but before published journal.
            self._append_event({"type": "published", "transaction": previous["transaction"]})
        self._append_event(reserve)
        try:
            self._install_payload(record["model"]["name"], model)
            self._install_payload(record["resume"]["name"], resume)
            self._fsync_directory()
            self._exclusive_write(record["latest_temp"], manifest)
            self._verify_file(record["latest_temp"], self._owned[record["latest_temp"]])
            # Refuse a changed/foreign latest rather than overwriting it.
            if self._latest_record(verify=True) != previous:
                raise CheckpointStoreError("latest changed during publication")
            self._path(_LATEST)
            self._path(record["latest_temp"])
            os.replace(record["latest_temp"], _LATEST, src_dir_fd=self._dirfd, dst_dir_fd=self._dirfd)
            self._fsync_directory()
        except BaseException:
            # Only exact, complete, still-owned temp files may be cleaned.
            # Partial writes/final payloads are left as auditable orphans.
            for name in (
                "." + record["model"]["name"] + ".tmp",
                "." + record["resume"]["name"] + ".tmp",
                record["latest_temp"],
            ):
                try:
                    identity = self._created.get(name)
                    if identity is None:
                        continue
                    info = os.stat(name, dir_fd=self._dirfd, follow_symlinks=False)
                    if (info.st_dev, info.st_ino) != identity:
                        continue
                    self._unlink_verified(name)
                    self._fsync_directory()
                except (OSError, CheckpointStoreError):
                    pass
            raise
        result = self._public_record(record)
        result["admission"] = {
            "free_bytes": free,
            "estimated_temporary_bytes": estimated,
            "min_free_bytes": self._owner["min_free_bytes"],
        }
        gc = {"deleted_paths": [], "deleted_bytes": 0, "errors": [], "orphan_paths": []}
        result["gc"] = gc
        try:
            self._append_event({"type": "published", "transaction": tx})
            self._collect(record, gc)
        except (OSError, CheckpointStoreError) as error:
            gc["errors"].append(str(error))
        return result

    def _collect(self, latest, gc):
        names = self._check_inventory(verify=False)
        history = sorted((self._records[t] for t in self._published), key=lambda r: r["update"], reverse=True)
        keep = {r["transaction"] for r in history[: self._owner["keep_last"]]}
        keep.add(latest["transaction"])
        # Do not discard a fallback unless all retained pairs are complete.
        for tx in keep:
            for kind in ("model", "resume"):
                item = self._records[tx][kind]
                self._verify_file(item["name"], (item["sha256"], item["size_bytes"]))
        victims = []
        for record in history:
            if record["transaction"] in keep:
                continue
            for kind in ("model", "resume"):
                if kind == "model" and record["archived_model"]:
                    continue
                if record[kind]["name"] in names:
                    victims.append(record[kind]["name"])
        if victims:
            self._append_event({"type": "gc", "names": victims})
            for name in victims:
                self._unlink_verified(name)
                gc["deleted_paths"].append(str(self.directory / name))
                gc["deleted_bytes"] += self._owned[name][1]
            self._fsync_directory()
        for tx, record in self._records.items():
            for name in self._reservation_files(record):
                is_temp = name.startswith(".")
                if (is_temp or tx not in self._published) and name in os.listdir(self._dirfd):
                    gc["orphan_paths"].append(str(self.directory / name))
        gc["kept_pair_updates"] = sorted(self._records[t]["update"] for t in keep)
        gc["archived_model_updates"] = sorted(r["update"] for r in history if r["archived_model"])

    def read_latest_bytes(self):
        """Return (verified model bytes, verified resume bytes, record) or None.

        Hash verification does not authenticate an untrusted pickle. Only the
        caller may unpickle these bytes, after establishing trusted provenance.
        """
        self._check_inventory(verify=False)
        record = self._latest_record(verify=True)
        if record is None:
            return None
        blobs = [
            self._verify_file(record[kind]["name"], (record[kind]["sha256"], record[kind]["size_bytes"]))
            for kind in ("model", "resume")
        ]
        return blobs[0], blobs[1], self._public_record(record)

    def close(self):
        for attr in ("_journal_fd", "_dirfd"):
            fd = getattr(self, attr, None)
            if fd is not None:
                os.close(fd)
                setattr(self, attr, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


# --- from vrpo_checkpoint.py ---


VRPO_RESUME_KIND = "vrpo_boundary_resume"
VRPO_RESUME_SCHEMA_VERSION = 3  # Legacy v2 CPU/reference protocol only.
VRPO_V5_RESUME_SCHEMA_VERSION = 5
VRPO_MODEL_KIND = "vrpo_model"
_CONTRACT_FIELDS = {
    "network_config",
    "head_config",
    "critic_config",
    "algorithm_config",
    "reward_config",
    "optimizer_config",
    "environment_config",
    "source_versions",
    "retention_enabled",
}
_NETWORK_FIELDS = {"architecture", "num_layers", "d_model", "num_heads", "mlp_dim"}
_ENVIRONMENT = {"env_name": "sanma", "round_mode": "half", "ruleset": RULESET}
_PROGRESS_FIELDS = {"completed_updates", "envsteps", "schedule_iteration"}
_STATE_FIELDS = {
    "params",
    "actor_opt_state",
    "critic_opt_state",
    "actor_steps",
    "critic_steps",
    "completed_updates",
}
_PAYLOAD_FIELDS = {
    "schema_version",
    "checkpoint_kind",
    "phase",
    "boundary",
    "config_contract",
    "train_state",
    "rng",
    "progress",
    "collection_state",
    "retention_state",
    "lineage",
}


@dataclass(frozen=True)
class VRPOResumeResult:
    """Fully checked, detached candidate; never mutates its template/runtime."""

    state: VRPOTrainingState
    rng: object
    progress: dict
    collection_state: object
    retention_state: object
    lineage: dict


def validate_vrpo_contract(contract):
    """Validate explicit complete schemas, retaining every provenance field.

    Algorithm/optimizer/source version mappings are supplied by the owner of the
    static training implementation; their complete types, keys and values must
    match exactly at restore, with no inferred fields or migration defaults.
    """
    _mapping(contract, "contract", required={"network_config"})
    network = _mapping(contract["network_config"], "network_config", required=_NETWORK_FIELDS, exact=True)
    config = validate_network_config(dict(network), "sanma")
    architecture = config["architecture"]
    if architecture not in ("v4", "v5"):
        raise ValueError("VRPO requires explicit network architecture v4 or v5")
    fields = _CONTRACT_FIELDS | ({"reading_config"} if architecture == "v5" else set())
    _mapping(contract, "contract", required=fields, exact=True)
    if contract["head_config"] is None:
        raise ValueError("head_config must be explicit, not inferred")
    validate_head_config(contract["head_config"], architecture)
    if architecture == "v5":
        if contract["reading_config"] is None:
            raise ValueError("V5 reading_config must be explicit, not inferred")
        validate_reading_config(contract["reading_config"], architecture)
    critic = validate_critic_config(contract["critic_config"])
    if critic != DEFAULT_CRITIC_CONFIG:
        raise ValueError("VRPO requires explicit central_action_q critic_config")
    reward = contract["reward_config"]
    if reward is None:
        raise ValueError("reward_config must be explicit, not inferred")
    validate_reward_config(reward, "sanma")
    environment = _copy_metadata(contract["environment_config"], "environment_config")
    _equal_contract(environment, _ENVIRONMENT, "environment_config")
    if type(contract["retention_enabled"]) is not bool:
        raise ValueError("retention_enabled must be bool")
    for name in ("algorithm_config", "optimizer_config", "source_versions"):
        if not _mapping(contract[name], name):
            raise ValueError(f"{name} must be explicit and nonempty")
    from .configs import ALGORITHM_KIND

    algorithm = contract["algorithm_config"]
    if architecture == "v5":
        validate_v5_reading_algorithm(algorithm)
    elif (
        type(algorithm.get("version")) is not int
        or algorithm["version"] != 2
        or algorithm.get("kind") != ALGORITHM_KIND
        or algorithm.get("reading_auxiliary", False) is not False
    ):
        raise ValueError("VRPO v2 algorithm contract required; no v1/V5 boundary migration")
    return _copy_metadata(contract, "contract")


def validate_v5_reading_algorithm(algorithm):
    """V5 never infers enabled reading or silently accepts an older protocol."""
    import math

    from .configs import READING_ALGORITHM_KIND, READING_PROTOCOL

    _mapping(
        algorithm,
        "algorithm_config",
        required={"version", "kind", "reading_auxiliary", "reading_aux_coef", "reading_protocol"},
    )
    if (
        type(algorithm["version"]) is not int
        or (algorithm["version"], algorithm["kind"]) != (5, READING_ALGORITHM_KIND)
        or algorithm["reading_auxiliary"] is not True
    ):
        raise ValueError("V5 requires explicit enabled version-5 reading algorithm")
    coef = algorithm["reading_aux_coef"]
    if isinstance(coef, bool) or not isinstance(coef, (int, float)) or not math.isfinite(coef) or coef <= 0:
        raise ValueError("V5 reading_aux_coef must be finite and positive")
    _equal_contract(algorithm["reading_protocol"], READING_PROTOCOL, "reading_protocol")


def _resume_schema(contract):
    return (
        VRPO_V5_RESUME_SCHEMA_VERSION
        if contract["network_config"]["architecture"] == "v5"
        else VRPO_RESUME_SCHEMA_VERSION
    )


def _container_tree(value, name):
    """Only transparent numeric containers, not arbitrary registered pytrees.

    A generic tree flatten alone would accidentally preserve TrainState static
    auxiliary fields inside a PyTreeDef. Explicit recursion rejects dataclasses,
    functions and transformation objects before any such reconstruction.
    """
    if isinstance(value, Mapping):
        if type(value) not in (dict, FrozenDict):
            raise ValueError(f"{name} contains unsupported mapping type")
        if not all(isinstance(key, str) and key for key in value):
            raise ValueError(f"{name} keys must be nonempty strings")
        for key, item in value.items():
            _container_tree(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        named = isinstance(value, tuple) and hasattr(value, "_fields")
        if type(value) not in (tuple, list) and not named:
            raise ValueError(f"{name} contains unsupported container type")
        for index, item in enumerate(value):
            _container_tree(item, f"{name}[{index}]")
    elif not isinstance(value, (jax.Array, np.ndarray, np.generic)):
        raise ValueError(f"{name} must contain arrays, not static objects or scalar leaves")


def _host_tree(value, name, *, prototype=None, optimizer=False):
    _container_tree(value, name)
    leaves, structure = jax.tree.flatten(value)
    if not leaves and not optimizer:
        raise ValueError(f"{name} must contain array leaves")
    if prototype is not None:
        _container_tree(prototype, f"template {name}")
        refs, expected_structure = jax.tree.flatten(prototype)
        if structure != expected_structure:
            raise ValueError(f"{name} tree structure mismatch")
    else:
        refs = [None] * len(leaves)
    arrays = []
    for index, (leaf, ref) in enumerate(zip(leaves, refs)):
        label = f"{name} leaf {index}"
        array = _checked_array(leaf, label)
        if np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(f"{label} must contain real numeric values")
        if ref is not None:
            expected = _checked_array(ref, f"template {label}")
            if array.shape != expected.shape or array.dtype != expected.dtype:
                raise ValueError(f"{label} shape/dtype mismatch")
        if optimizer and np.issubdtype(array.dtype, np.integer) and np.any(array < 0):
            raise ValueError(f"{label} has a negative optimizer counter")
        arrays.append(array.copy())
    return jax.tree.unflatten(structure, arrays)


def _state_counter(value, name, prototype=None):
    array = _checked_array(value, name)
    if array.shape != () or not np.issubdtype(array.dtype, np.signedinteger) or int(array) < 0:
        raise ValueError(f"{name} must be a nonnegative scalar signed integer array")
    if prototype is not None:
        expected = _checked_array(prototype, f"template {name}")
        if expected.shape != () or array.dtype != expected.dtype:
            raise ValueError(f"{name} shape/dtype mismatch")
    return array.copy()


def _progress(value):
    _mapping(value, "progress", required=_PROGRESS_FIELDS, exact=True)
    progress = {key: _counter(value[key], f"progress.{key}") for key in sorted(_PROGRESS_FIELDS)}
    if progress["schedule_iteration"] != progress["completed_updates"]:
        raise ValueError("progress.schedule_iteration must match progress.completed_updates")
    return progress


def _runtime(collection_state, retention_state, lineage, contract, progress):
    games = contract["algorithm_config"].get("rollout_games")
    if type(games) is not int or games <= 0:
        raise ValueError("explicit rollout_games required for collection boundary clock")
    collection = checked_collection_state(collection_state, expected_next_game_id=progress["completed_updates"] * games)
    if contract["retention_enabled"] and retention_state is None:
        raise ValueError("enabled retention requires complete retention_state")
    if not contract["retention_enabled"] and retention_state is not None:
        raise ValueError("disabled retention requires retention_state=None")
    lineage = {} if lineage is None else lineage
    _mapping(lineage, "lineage")
    return collection, _copy_metadata(retention_state, "retention_state"), _copy_metadata(lineage, "lineage")


def _export_state(state):
    if type(state) is not VRPOTrainingState:
        raise ValueError("state must be the dynamic-only VRPOTrainingState")
    return {
        "params": _host_tree(state.params, "params"),
        "actor_opt_state": _host_tree(state.actor_opt_state, "actor_opt_state", optimizer=True),
        "critic_opt_state": _host_tree(state.critic_opt_state, "critic_opt_state", optimizer=True),
        **{
            key: _state_counter(getattr(state, key), key)
            for key in ("actor_steps", "critic_steps", "completed_updates")
        },
    }


def _device_tree(value, name):
    def convert(array):
        converted = jnp.asarray(array)
        if converted.dtype != array.dtype or converted.shape != array.shape:
            raise ValueError(f"{name} cannot be restored without dtype/shape conversion")
        return converted

    return jax.tree.map(convert, value)


def build_vrpo_resume(state, rng, *, contract, progress, collection_state, retention_state=None, lineage=None):
    """Export one fully committed complete update as detached HOST dynamic data.

    The caller guarantees all games are drained, with no active collector/batch
    and no unfinished Actor/Critic optimization phase. There is no partial-update
    restore path or conversion from a legacy PPO boundary schema.
    """
    saved_contract = validate_vrpo_contract(contract)
    saved_progress = _progress(progress)
    dynamic = _export_state(state)
    if int(dynamic["completed_updates"]) != saved_progress["completed_updates"]:
        raise ValueError("state.completed_updates must match progress.completed_updates")
    collection, retention, provenance = _runtime(
        collection_state, retention_state, lineage, saved_contract, saved_progress
    )
    return {
        "checkpoint_kind": VRPO_RESUME_KIND,
        "schema_version": _resume_schema(saved_contract),
        "phase": "complete",
        "boundary": {"kind": "update_boundary", "active_batch": False},
        "config_contract": saved_contract,
        "train_state": dynamic,
        "rng": _rng(rng),
        "progress": saved_progress,
        "collection_state": collection,
        "retention_state": retention,
        "lineage": provenance,
    }


def restore_vrpo_resume(payload, template_state, *, contract):
    """Validate EVERYTHING before returning an independent immutable candidate.

    Both optimizer trees are restored exactly, not reconstructed as fresh slots.
    The collection clock and optional retention are checked before returning any
    candidate. No assignment to a supplied object occurs here.
    """
    expected = validate_vrpo_contract(contract)
    _mapping(payload, "VRPO resume payload", required={"schema_version", "checkpoint_kind"})
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != _resume_schema(expected)
        or payload["checkpoint_kind"] != VRPO_RESUME_KIND
    ):
        raise ValueError("unsupported VRPO boundary schema/version/kind (no legacy migration)")
    _mapping(payload, "VRPO resume payload", required=_PAYLOAD_FIELDS, exact=True)
    if payload["phase"] != "complete":
        raise ValueError("VRPO boundary restore requires phase='complete'")
    boundary = _mapping(payload["boundary"], "boundary", required={"kind", "active_batch"}, exact=True)
    if boundary["kind"] != "update_boundary" or boundary["active_batch"] is not False:
        raise ValueError("VRPO restore requires update_boundary with no active batch")
    actual = validate_vrpo_contract(payload["config_contract"])
    _equal_contract(actual, expected, "config_contract")
    if type(template_state) is not VRPOTrainingState:
        raise ValueError("template_state must be the dynamic-only VRPOTrainingState")
    dynamic = _mapping(payload["train_state"], "train_state", required=_STATE_FIELDS, exact=True)
    restored = {
        "params": _host_tree(dynamic["params"], "params", prototype=template_state.params),
        "actor_opt_state": _host_tree(
            dynamic["actor_opt_state"], "actor_opt_state", prototype=template_state.actor_opt_state, optimizer=True
        ),
        "critic_opt_state": _host_tree(
            dynamic["critic_opt_state"], "critic_opt_state", prototype=template_state.critic_opt_state, optimizer=True
        ),
        **{
            key: _state_counter(dynamic[key], key, getattr(template_state, key))
            for key in ("actor_steps", "critic_steps", "completed_updates")
        },
    }
    rng = _rng(payload["rng"])
    progress = _progress(payload["progress"])
    if int(restored["completed_updates"]) != progress["completed_updates"]:
        raise ValueError("state.completed_updates must match progress.completed_updates")
    collection, retention, lineage = _runtime(
        payload["collection_state"], payload["retention_state"], payload["lineage"], expected, progress
    )
    # Only create a fresh state after every schema/data/runtime check succeeds.
    state = template_state.replace(**{key: _device_tree(value, key) for key, value in restored.items()})
    return VRPOResumeResult(state, jnp.asarray(rng), progress, collection, retention, lineage)


def build_vrpo_model(params, *, contract, metadata=None):
    """Ordinary model initialization artifact: NO optimizer/RNG/full-resume state."""
    config = validate_vrpo_contract(contract)
    metadata = {} if metadata is None else _copy_metadata(metadata, "metadata")
    _mapping(metadata, "metadata")
    for key, required in (("optimizer_state_included", False), ("resume_semantics", "initialization_only")):
        if key in metadata and (type(metadata[key]) is not type(required) or metadata[key] != required):
            raise ValueError(f"ordinary vrpo_model metadata.{key} must be {required!r}")
        metadata[key] = required
    return {
        "checkpoint_kind": VRPO_MODEL_KIND,
        "schema_version": 1,
        **{
            key: config[key]
            for key in (
                "network_config",
                "head_config",
                "critic_config",
                "algorithm_config",
                "reward_config",
                "source_versions",
            )
        },
        **(
            {key: config[key] for key in ("reading_config", "optimizer_config")}
            if config["network_config"]["architecture"] == "v5"
            else {}
        ),
        **config["environment_config"],
        "params": _host_tree(params, "params"),
        "metadata": metadata,
    }
