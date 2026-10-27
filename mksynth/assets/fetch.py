"""Download assets listed in configs/assets.yaml and record their licences.

Each asset lands in ``<raw_root>/<id>/`` with the upstream files plus ``source.json``
(source, upstream name, licence, attribution, URL, SHA-256 of every file). Downloads are
untrusted data: they are only parsed as meshes/images, never executed or imported.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mksynth.config import REPO_ROOT, load_yaml, resolve

FUEL_API = "https://fuel.gazebosim.org/1.0/{owner}/models/{name}"
LICENCE_IDS = {
    "Creative Commons Attribution 4.0 International": "CC-BY-4.0",
    "Creative Commons Zero v1.0 Universal": "CC0-1.0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_get(url: str, timeout: float = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "messykitchens-synth/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Refusing archive member outside destination: {member.filename}")
    archive.extractall(destination)


def fetch_gazebo_fuel(asset: dict, source_cfg: dict, destination: Path) -> dict:
    owner = source_cfg["owner"]
    meta = json.loads(_http_get(FUEL_API.format(owner=owner, name=asset["name"])))
    licence = LICENCE_IDS.get(meta.get("license_name"), meta.get("license_name"))
    payload = _http_get(source_cfg["url"].format(name=asset["name"]))
    staging = destination.with_name(destination.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        _safe_extract(archive, staging)
    if not (staging / "meshes" / "model.obj").is_file():
        raise FileNotFoundError(f"{asset['id']}: archive has no meshes/model.obj")
    return {
        "licence": licence,
        "licence_url": meta.get("license_url"),
        "upstream_name": meta.get("name"),
        "upstream_owner": owner,
        "upstream_modified": meta.get("modify_date"),
        "url": source_cfg["url"].format(name=asset["name"]),
        "staging": staging,
    }


FETCHERS = {"gazebo_fuel": fetch_gazebo_fuel}


def fetch_asset(asset: dict, assets_cfg: dict, raw_root: Path, force: bool = False) -> dict:
    destination = raw_root / asset["id"]
    record_path = destination / "source.json"
    if record_path.is_file() and not force:
        return json.loads(record_path.read_text())
    source_cfg = assets_cfg["sources"][asset["source"]]
    result = FETCHERS[source_cfg["fetcher"]](asset, source_cfg, destination)
    staging: Path = result.pop("staging")
    files = {
        str(path.relative_to(staging)): sha256_file(path)
        for path in sorted(staging.rglob("*")) if path.is_file()
    }
    record = {
        "id": asset["id"],
        "source": asset["source"],
        "name": asset["name"],
        "attribution": source_cfg.get("attribution", ""),
        **result,
        "mesh": "meshes/model.obj",
        "files_sha256": files,
    }
    (staging / "source.json").write_text(json.dumps(record, indent=2) + "\n")
    shutil.rmtree(destination, ignore_errors=True)
    staging.rename(destination)
    return record


def load_holdout_hashes(assets_cfg: dict) -> set[str]:
    holdout = assets_cfg.get("holdout", {})
    listing = holdout.get("sha256_list")
    if not listing:
        return set()
    entries = json.loads(resolve(listing).read_text())
    hashes = set()
    for entry in entries:
        lfs = entry.get("lfs") or {}
        if lfs.get("oid"):
            hashes.add(lfs["oid"])
    return hashes


def check_asset_allowed(record: dict, assets_cfg: dict, holdout_hashes: set[str]) -> None:
    """Raise if an asset violates the licence rule or the benchmark hold-out rule."""
    if record["licence"] not in assets_cfg["allowed_licences"]:
        raise ValueError(f"{record['id']}: licence {record['licence']!r} is not allowed")
    if record["source"] in set(assets_cfg.get("holdout", {}).get("sources", [])):
        raise ValueError(f"{record['id']}: source {record['source']!r} is a held-out benchmark source")
    clash = holdout_hashes.intersection(record["files_sha256"].values())
    if clash:
        raise ValueError(f"{record['id']}: file hash matches a MessyKitchens benchmark object")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", default=str(REPO_ROOT / "configs/assets.yaml"))
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "data/assets/raw"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--only", nargs="*", help="Asset ids to fetch (default: all in --splits)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    assets_cfg = load_yaml(args.assets)
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    holdout_hashes = load_holdout_hashes(assets_cfg)
    selected = [
        asset for asset in assets_cfg["assets"]
        if (args.only and asset["id"] in args.only) or (not args.only and asset["split"] in args.splits)
    ]

    def work(asset: dict) -> tuple[str, str]:
        try:
            record = fetch_asset(asset, assets_cfg, raw_root, force=args.force)
            check_asset_allowed(record, assets_cfg, holdout_hashes)
            return asset["id"], f"ok ({record['licence']})"
        except Exception as error:  # noqa: BLE001 - report every asset
            return asset["id"], f"FAILED: {type(error).__name__}: {error}"

    with ThreadPoolExecutor(args.workers) as pool:
        for asset_id, status in pool.map(work, selected):
            print(f"{asset_id:32s} {status}")


if __name__ == "__main__":
    main()
