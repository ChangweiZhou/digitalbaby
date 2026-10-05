#!/usr/bin/env python3
"""Verify and restore the public ERROR scientific supplement (Python 3.9+).

SPDX-License-Identifier: GPL-3.0-or-later
No network access, credentials, or third-party packages are used.
"""
import argparse
import hashlib
import io
import json
import lzma
import pathlib
import sys
import tarfile

PREFIX = "studies/error-completion-20261004/"
BLOCK = 65536


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(value):
    require(isinstance(value, str) and value and "\\" not in value,
            "Invalid relative path")
    p = pathlib.PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and
            str(p) == value, "Unsafe or noncanonical relative path")
    return p


def checked_file(root, entry, maximum):
    rel = safe_path(entry["path"])
    p = root.joinpath(*rel.parts)
    require(p.is_file() and not p.is_symlink() and
            p.resolve().is_relative_to(root), "Missing or unsafe bundle file: " + str(rel))
    size = entry["bytes"]
    require(isinstance(size, int) and 0 <= size <= maximum,
            "Invalid bundle file size")
    require(p.stat().st_size == size, "Size mismatch: " + str(rel))
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(BLOCK), b""):
            h.update(b)
    require(h.hexdigest() == entry["sha256"], "SHA256 mismatch: " + str(rel))
    return p


class JoinedChunks(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    return 0
            n = self.current.readinto(b)
            if n:
                return n
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


class BoundedXZ(io.RawIOBase):
    def __init__(self, source, expected_bytes):
        self.source = source
        self.decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ,
                                            memlimit=128 * 1024 * 1024)
        self.pending = b""
        self.total = 0
        self.expected = expected_bytes
        self.done = False

    def readable(self):
        return True

    def readinto(self, b):
        while not self.pending:
            if self.done:
                return 0
            if self.decoder.eof:
                require(not self.decoder.unused_data and not self.source.read(1),
                        "Unexpected trailing compressed data")
                require(self.total == self.expected, "Uncompressed archive size mismatch")
                self.done = True
                return 0
            raw = self.source.read(BLOCK) if self.decoder.needs_input else b""
            require(raw or not self.decoder.needs_input, "Truncated XZ stream")
            self.pending = self.decoder.decompress(raw, max_length=BLOCK)
            self.total += len(self.pending)
            require(self.total <= self.expected, "Uncompressed archive exceeds manifest size")
        n = min(len(b), len(self.pending))
        b[:n] = self.pending[:n]
        self.pending = self.pending[n:]
        return n


def verify_or_extract(paths, expected, tar_bytes, output=None):
    seen = set()
    with JoinedChunks(paths) as joined:
        with io.BufferedReader(joined, buffer_size=BLOCK) as packed:
            decoder = BoundedXZ(packed, tar_bytes)
            with io.BufferedReader(decoder, buffer_size=BLOCK) as raw:
                with tarfile.open(fileobj=raw, mode="r|") as archive:
                    for member in archive:
                        name = member.name
                        rel = safe_path(name)
                        require(name.startswith(PREFIX) and name in expected,
                                "Unexpected archive path: " + name)
                        require(name not in seen and member.isfile(),
                                "Duplicate or nonregular archive entry: " + name)
                        entry = expected[name]
                        require(member.size == entry["bytes"], "Archive member size mismatch: " + name)
                        target = None
                        if output is not None:
                            target_path = output.joinpath(*rel.parts)
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            require(target_path.parent.resolve().is_relative_to(output),
                                    "Unsafe output parent")
                            target = target_path.open("xb")
                        h = hashlib.sha256()
                        count = 0
                        try:
                            with archive.extractfile(member) as source:
                                for part in iter(lambda: source.read(BLOCK), b""):
                                    count += len(part)
                                    h.update(part)
                                    if target is not None:
                                        target.write(part)
                        finally:
                            if target is not None:
                                target.close()
                        require(count == entry["bytes"] and h.hexdigest() == entry["sha256"],
                                "Restored byte identity mismatch: " + name)
                        seen.add(name)
                # Force the XZ checksum/end marker and exact decompressed length to be checked.
                for padding in iter(lambda: raw.read(BLOCK), b""):
                    require(not any(padding), "Nonzero data after the tar end marker")
    require(seen == set(expected), "Missing archive members")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent)
    parser.add_argument("--output", type=pathlib.Path,
                        help="New, nonexistent directory for the restored supplement")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    require(args.verify_only != (args.output is not None),
            "Choose exactly one of --verify-only or --output")
    root = args.bundle_dir.resolve()
    mp = root / "BUNDLE_MANIFEST.json"
    require(mp.is_file() and not mp.is_symlink() and mp.stat().st_size < 131072,
            "Missing or oversized bundle manifest")
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    require(manifest["schema"] == "ERROR-PUBLIC-SCIENTIFIC-BUNDLE-v1" and
            manifest["archive_format"] == "tar.xz" and manifest["file_count"] == 8453,
            "Unsupported bundle manifest")
    expected = {}
    for part in manifest["file_index_parts"]:
        p = checked_file(root, part, 262144)
        records = json.loads(p.read_text(encoding="utf-8"))
        require(isinstance(records, list), "Invalid file index")
        for entry in records:
            name = str(safe_path(entry["path"]))
            require(name.startswith(PREFIX) and name not in expected,
                    "Duplicate or out-of-scope file index entry")
            require(isinstance(entry["bytes"], int) and entry["bytes"] >= 0,
                    "Invalid decoded file size")
            expected[name] = entry
    require(len(expected) == manifest["file_count"] and
            sum(e["bytes"] for e in expected.values()) == manifest["source_bytes"],
            "File index count or byte total mismatch")
    paths = []
    total = 0
    archive_hash = hashlib.sha256()
    for i, chunk in enumerate(manifest["chunks"]):
        require(chunk["path"] == f"chunks/scientific-files.tar.xz.part{i:03d}",
                "Invalid chunk order or filename")
        p = checked_file(root, chunk, 393216)
        paths.append(p)
        total += chunk["bytes"]
        with p.open("rb") as source:
            for b in iter(lambda: source.read(BLOCK), b""):
                archive_hash.update(b)
    require(total == manifest["archive_bytes"] and
            archive_hash.hexdigest() == manifest["archive_sha256"],
            "Combined archive identity mismatch")
    verify_or_extract(paths, expected, manifest["tar_bytes"])
    output = None
    if args.output is not None:
        require(not args.output.exists() and not args.output.is_symlink(),
                "Output must not already exist")
        args.output.mkdir(parents=False, exist_ok=False)
        output = args.output.resolve()
        verify_or_extract(paths, expected, manifest["tar_bytes"], output)
    print(json.dumps({"verified": True, "files": len(expected),
                      "bytes": manifest["source_bytes"],
                      "archive_sha256": manifest["archive_sha256"],
                      "output": str(output) if output is not None else None}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, lzma.LZMAError, tarfile.TarError) as exc:
        print("Bundle verification failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
