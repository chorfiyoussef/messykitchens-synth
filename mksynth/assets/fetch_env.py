"""Download lighting HDRIs and table/background materials from Poly Haven (CC0).

Selection is deterministic: the most-downloaded assets of the configured categories at the
time of the first run. The chosen ids, file hashes and licence are written to
``environment.json``; later runs reuse that list so the environment pool stays fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mksynth.config import REPO_ROOT

API = "https://api.polyhaven.com"
HEADERS = {"User-Agent": "messykitchens-synth/0.1 (synthetic data research)"}
TEXTURE_MAPS = {"diffuse": "Diffuse", "roughness": "Rough", "normal": "nor_gl"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as response:
        return response.read()


def _pick(kind: str, categories: list[str], count: int, exclude: list[str]) -> list[str]:
    assets = json.loads(_get(f"{API}/assets?t={kind}&c={','.join(categories)}"))
    ranked = sorted(assets.items(), key=lambda item: (-item[1].get("download_count", 0), item[0]))
    return [asset_id for asset_id, info in ranked
            if not any(word in " ".join(info.get("tags", []) + info.get("categories", [])) for word in exclude)][:count]


def _download(url: str, destination: Path) -> str:
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_get(url))
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def fetch(out_dir: Path, n_hdris: int, n_textures: int, resolution: str) -> dict:
    manifest_path = out_dir / "environment.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        hdri_ids = [h["id"] for h in previous["hdris"]]
        texture_ids = [t["id"] for t in previous["materials"]]
    else:
        hdri_ids = _pick("hdris", ["indoor"], n_hdris, exclude=["night"])
        texture_ids = []
        for category, share in (("wood", 0.4), ("fabric", 0.2), ("plaster-concrete", 0.2), ("tiles", 0.2)):
            picked = _pick("textures", [category], n_textures, exclude=["terrain", "outdoor"])
            texture_ids += [t for t in picked if t not in texture_ids][: max(1, round(share * n_textures))]

    def get_hdri(asset_id: str) -> dict:
        files = json.loads(_get(f"{API}/files/{asset_id}"))
        entry = files["hdri"][resolution]["hdr"]
        path = out_dir / "hdris" / f"{asset_id}_{resolution}.hdr"
        digest = _download(entry["url"], path)
        return {"id": asset_id, "path": str(path.relative_to(REPO_ROOT)), "sha256": digest, "licence": "CC0-1.0",
                "url": f"https://polyhaven.com/a/{asset_id}"}

    def get_texture(asset_id: str) -> dict:
        files = json.loads(_get(f"{API}/files/{asset_id}"))
        maps = {}
        for name, key in TEXTURE_MAPS.items():
            if key not in files:
                continue
            entry = files[key][resolution]["jpg" if "jpg" in files[key][resolution] else "png"]
            path = out_dir / "materials" / asset_id / Path(entry["url"]).name
            maps[name] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": _download(entry["url"], path)}
        return {"id": asset_id, "maps": maps, "licence": "CC0-1.0", "url": f"https://polyhaven.com/a/{asset_id}"}

    with ThreadPoolExecutor(8) as pool:
        hdris = list(pool.map(get_hdri, hdri_ids))
        materials = [m for m in pool.map(get_texture, texture_ids) if "diffuse" in m["maps"]]
    manifest = {"source": "Poly Haven (polyhaven.com)", "licence": "CC0-1.0", "resolution": resolution,
                "hdris": hdris, "materials": materials}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data/environment"))
    parser.add_argument("--hdris", type=int, default=30)
    parser.add_argument("--textures", type=int, default=30)
    parser.add_argument("--resolution", default="1k")
    args = parser.parse_args()
    manifest = fetch(Path(args.out_dir), args.hdris, args.textures, args.resolution)
    print(f"{len(manifest['hdris'])} HDRIs, {len(manifest['materials'])} materials -> {args.out_dir}/environment.json")


if __name__ == "__main__":
    main()
