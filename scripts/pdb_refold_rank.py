#!/usr/bin/env python3
"""Refold target-binder PDB pairs with Boltz-2 and rank using BoltzGen-style metrics.

This script is intentionally stand-alone for non-BoltzGen input lists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}

POSITIVE_RES = {"ARG", "LYS", "HIS"}
NEGATIVE_RES = {"ASP", "GLU"}


@dataclass
class Entry:
    pdb_id: str
    chain_1: str
    chain_2: str


@dataclass
class RefoldResult:
    entry: Entry
    pdb_path: Path
    refold_cif: Path
    confidence_json: Path
    design_iiptm: float
    design_ptm: float
    min_design_to_target_pae: float
    neg_min_design_to_target_pae: float
    delta_sasa_refolded: float
    design_sasa_unbound_refolded: float
    design_sasa_bound_refolded: float
    plip_hbonds_refolded: int
    plip_saltbridge_refolded: int


def maybe_install(requirements: Sequence[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--workdir", default="boltzgen_refold_work")
    p.add_argument("--pdb_dir", default=None, help="Optional local cache for source PDB files")
    p.add_argument("--install_requirements", action="store_true")
    p.add_argument(
        "--boltz_binary",
        default="boltz",
        help="Boltz CLI executable name/path used for `boltz predict`",
    )
    p.add_argument(
        "--boltz_predict_extra_args",
        nargs="*",
        default=[],
        help="Extra args appended to `boltz predict`",
    )
    p.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute backend forwarded to `boltz predict --accelerator`.",
    )
    p.add_argument("--sampling_steps", type=int, default=200)
    p.add_argument("--diffusion_samples", type=int, default=5)
    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument(
        "--boltz_cache_dir",
        default=None,
        help="Cache directory passed to `boltz predict --cache`. Defaults to `<workdir>/boltz_cache`.",
    )
    p.add_argument(
        "--use_msa_server",
        action="store_true",
        help="Pass `--use_msa_server` through to `boltz predict` when MSA generation is required.",
    )
    p.add_argument("--keep_intermediates", action="store_true")
    p.add_argument(
        "--validate_only",
        action="store_true",
        help="Only download/parse PDB files and validate chain IDs without running `boltz predict`.",
    )
    return p.parse_args()


def read_entries(csv_path: Path) -> List[Entry]:
    df = pd.read_csv(csv_path)
    expected = ["PDB ID", "Chain_1", "Chain_2"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected columns: {expected}")
    return [
        Entry(str(r["PDB ID"]).strip().lower(), str(r["Chain_1"]).strip(), str(r["Chain_2"]).strip())
        for _, r in df.iterrows()
    ]


def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pdb_id}.pdb"
    if out.exists():
        return out
    import urllib.request

    urls = [
        f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
        f"https://models.rcsb.org/{pdb_id.lower()}.pdb",
    ]
    last_exc = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "boltzgen-pdb-refold-rank/1.0"})
            with urllib.request.urlopen(req) as r:
                out.write_bytes(r.read())
            if out.stat().st_size == 0:
                raise ValueError(f"Downloaded empty file from {url}")
            return out
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to download {pdb_id} from known endpoints: {last_exc}")
    return out


def parse_structure(pdb_path: Path):
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_path.stem, str(pdb_path))


def extract_chain_sequence(structure, chain_id: str) -> str:
    chain = None
    available_chain_ids = []
    search_ids = [chain_id, chain_id.upper(), chain_id.lower()]
    for model in structure:
        available_chain_ids = [c.id for c in model]
        chain = next((model[cid] for cid in search_ids if cid in model), None)
        if chain is not None:
            break
    if chain is None:
        raise ValueError(f"Chain {chain_id} not found. Available chains: {available_chain_ids}")

    seq = []
    seen = set()
    for res in chain:
        if res.id[0] != " ":
            continue
        key = (res.id[1], res.id[2])
        if key in seen:
            continue
        seen.add(key)
        seq.append(AA3_TO_1.get(res.resname.upper(), "X"))
    return "".join(seq)


def write_fasta_for_boltz(target_seq: str, binder_seq: str, out_fa: Path) -> None:
    # Boltz FASTA headers use the first field as the chain ID. Keep IDs short
    # and stable because downstream Boltz metadata stores chain names in a
    # fixed-width field; descriptive IDs like "target" can be truncated.
    out_fa.write_text(f">A|protein\n{target_seq}\n>B|protein\n{binder_seq}\n")


def run_boltz_predict(
    boltz_binary: str,
    fasta_path: Path,
    output_dir: Path,
    extra_args: Sequence[str],
    accelerator: str,
    sampling_steps: int,
    diffusion_samples: int,
    recycling_steps: int,
    use_msa_server: bool,
    cache_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        boltz_binary,
        "predict",
        str(fasta_path),
        "--out_dir",
        str(output_dir),
        "--cache",
        str(cache_dir),
        "--sampling_steps",
        str(sampling_steps),
        "--diffusion_samples",
        str(diffusion_samples),
        "--recycling_steps",
        str(recycling_steps),
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    if accelerator != "auto":
        cmd.extend(["--accelerator", accelerator])
    cmd.extend(extra_args)
    cache_root = output_dir / ".runtime_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    env.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    subprocess.check_call(cmd, env=env)


def _list_confidence_jsons(root: Path) -> List[Path]:
    return sorted(root.rglob("confidence*.json"))


def _safe_get(d: dict, key: str, default=np.nan):
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


def choose_best_sample_and_metrics(boltz_out_dir: Path) -> Tuple[Path, Path, Dict[str, float]]:
    confs = _list_confidence_jsons(boltz_out_dir)
    if not confs:
        msg = f"No confidence JSON found under {boltz_out_dir}"
        processed_inputs = sorted(boltz_out_dir.rglob("processed/*"))
        if processed_inputs:
            msg += f". Processed inputs present: {[p.name for p in processed_inputs]}"
        raise FileNotFoundError(msg)

    best_score = -math.inf
    best_payload = None
    best_conf_path = None
    for conf_path in confs:
        payload = json.loads(conf_path.read_text())
        design_to_target_iptm = _safe_get(payload, "design_to_target_iptm", np.nan)
        design_ptm = _safe_get(payload, "design_ptm", np.nan)
        score = 0.8 * design_to_target_iptm + 0.2 * design_ptm
        if score > best_score:
            best_score = score
            best_payload = payload
            best_conf_path = conf_path

    assert best_payload is not None and best_conf_path is not None
    candidate_cifs = sorted(best_conf_path.parent.glob("*.cif")) + sorted(best_conf_path.parent.glob("*.mmcif"))
    if not candidate_cifs:
        candidate_cifs = sorted(boltz_out_dir.rglob("*.cif"))
    if not candidate_cifs:
        raise FileNotFoundError(f"Could not find refolded CIF for {best_conf_path}")

    return candidate_cifs[0], best_conf_path, best_payload


def _load_atoms(cif_path: Path):
    import biotite.structure.io.pdbx as pdbx

    cf = pdbx.CIFFile.read(str(cif_path))
    st = pdbx.get_structure(cf, use_author_fields=False)
    return st[0]


def _decode(v) -> str:
    return v.decode().strip() if isinstance(v, (bytes, bytearray)) else str(v).strip()


def _chain_mask(atoms, chain_id: str):
    ids = np.array([_decode(c) for c in atoms.chain_id])
    return ids == chain_id


def _sequence_from_atoms_chain(atoms, chain_id: str) -> str:
    mask = _chain_mask(atoms, chain_id)
    if not mask.any():
        return ""
    chain_atoms = atoms[mask]
    seq: List[str] = []
    seen = set()
    for res_name, res_id in zip(chain_atoms.res_name, chain_atoms.res_id):
        key = int(res_id)
        if key in seen:
            continue
        seen.add(key)
        seq.append(AA3_TO_1.get(_decode(res_name).upper(), "X"))
    return "".join(seq)


def detect_refold_chain_mapping(refold_cif: Path, target_seq: str, binder_seq: str) -> Tuple[str, str]:
    atoms = _load_atoms(refold_cif)
    chain_ids = sorted(set(_decode(c) for c in atoms.chain_id))
    if len(chain_ids) < 2:
        raise ValueError(f"Expected >=2 chains in refolded CIF, found {chain_ids}")

    best = None
    for t in chain_ids:
        for b in chain_ids:
            if t == b:
                continue
            t_seq = _sequence_from_atoms_chain(atoms, t)
            b_seq = _sequence_from_atoms_chain(atoms, b)
            if not t_seq or not b_seq:
                continue
            score = int(t_seq == target_seq) + int(b_seq == binder_seq)
            if best is None or score > best[0]:
                best = (score, t, b)

    if best is None:
        return "A", "B"
    _, target_chain, binder_chain = best
    return target_chain, binder_chain


def compute_delta_sasa_refolded(cif_path: Path, target_chain: str, binder_chain: str):
    from biotite.structure.info import vdw_radius_protor, vdw_radius_single
    from biotite.structure.sasa import sasa

    atoms = _load_atoms(cif_path)
    res = [_decode(r) for r in atoms.res_name]
    atm = [_decode(a) for a in atoms.atom_name]
    elem = [_decode(e) for e in atoms.element]

    def rad(rn: str, an: str, el: str) -> float:
        try:
            x = vdw_radius_protor(rn.upper(), an.upper())
            if x is None:
                x = vdw_radius_single(el)
            return float(x if x is not None else 1.5)
        except Exception:
            return 1.5

    radii = np.array([rad(rn, an, el) for rn, an, el in zip(res, atm, elem)], dtype=float)

    target_mask = _chain_mask(atoms, target_chain)
    binder_mask = _chain_mask(atoms, binder_chain)
    bound_mask = target_mask | binder_mask

    atoms_bound = atoms[bound_mask]
    area_bound = sasa(atoms_bound, probe_radius=1.4, point_number=960, vdw_radii=radii[bound_mask])
    target_bound = float(area_bound[target_mask[bound_mask]].sum())

    target_atoms = atoms[target_mask]
    target_area = sasa(target_atoms, probe_radius=1.4, point_number=960, vdw_radii=radii[target_mask])
    target_unbound = float(target_area.sum())
    return target_unbound - target_bound, target_unbound, target_bound


def count_refolded_noncovalents(cif_path: Path, binder_chain: str) -> Tuple[int, int]:
    import biotite
    import hydride
    import torch

    atoms = _load_atoms(cif_path)
    atoms_h, _ = hydride.add_hydrogen(atoms)

    chain_ids_h = np.array([_decode(c) for c in atoms_h.chain_id])
    is_design_h = chain_ids_h == binder_chain

    hbond = biotite.structure.hbond(atoms_h)
    if len(hbond) == 0:
        hbonds = 0
    else:
        donor_idxs, acceptor_idxs = hbond[:, 0], hbond[:, 2]
        donor_design_hbonds = int((is_design_h[donor_idxs] & ~is_design_h[acceptor_idxs]).sum())
        acceptor_design_hbonds = int((~is_design_h[donor_idxs] & is_design_h[acceptor_idxs]).sum())
        hbonds = donor_design_hbonds + acceptor_design_hbonds

    res_name_h = np.array([_decode(r).upper() for r in atoms_h.res_name])
    atom_name_h = np.array([_decode(a).upper() for a in atoms_h.atom_name])
    pos_mask = np.isin(res_name_h, list(POSITIVE_RES)) & np.isin(atom_name_h, ["NZ", "NH1", "NH2", "NE", "ND1", "NE2"])
    neg_mask = np.isin(res_name_h, list(NEGATIVE_RES)) & np.isin(atom_name_h, ["OD1", "OD2", "OE1", "OE2"])
    pos_atoms = atoms_h[pos_mask]
    neg_atoms = atoms_h[neg_mask]
    pos_chain = chain_ids_h[pos_mask]
    neg_chain = chain_ids_h[neg_mask]

    if len(pos_atoms) == 0 or len(neg_atoms) == 0:
        return hbonds, 0

    dmat = torch.cdist(torch.as_tensor(pos_atoms.coord), torch.as_tensor(neg_atoms.coord))
    pi, ni = torch.where((dmat > 0.5) & (dmat < 5.5))
    pi, ni = pi.cpu().numpy(), ni.cpu().numpy()

    pos_is_design = pos_chain == binder_chain
    neg_is_design = neg_chain == binder_chain
    pos_design_sb = int((pos_is_design[pi] & ~neg_is_design[ni]).sum())
    neg_design_sb = int((~pos_is_design[pi] & neg_is_design[ni]).sum())
    return hbonds, (pos_design_sb + neg_design_sb)


def rank_like_boltzgen(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_filters_passed"] = 1
    metrics = {
        "design_iiptm": 1,
        "design_ptm": 2,
        "neg_min_design_to_target_pae": 1,
        "plip_hbonds_refolded": 2,
        "plip_saltbridge_refolded": 2,
        "delta_sasa_refolded": 2,
    }
    rank_df = pd.DataFrame(index=df.index)
    for col, inverse_importance in metrics.items():
        rank_df[f"rank_{col}"] = (
            df[["num_filters_passed", col]].apply(tuple, axis=1).rank(method="min", ascending=False).astype(int)
            / inverse_importance
        )
    df = pd.concat([df, rank_df], axis=1)
    df["max_rank"] = rank_df.max(axis=1)
    df = df.sort_values("max_rank")
    df["secondary_rank"] = df["max_rank"].rank(method="dense").astype(int)
    df = df.sort_values(["secondary_rank", "design_iiptm"], ascending=[True, False])
    df["final_rank"] = np.arange(1, len(df) + 1)
    df["quality_score"] = 1.0 if len(df) == 1 else 1 - (df["final_rank"] - 1) / (len(df) - 1)
    return df


def process_entry(
    entry: Entry,
    pdb_dir: Path,
    workdir: Path,
    boltz_binary: str,
    boltz_predict_extra_args: Sequence[str],
    accelerator: str,
    sampling_steps: int,
    diffusion_samples: int,
    recycling_steps: int,
    use_msa_server: bool,
    boltz_cache_dir: Path,
) -> RefoldResult:
    pdb_path = download_pdb(entry.pdb_id, pdb_dir)
    structure = parse_structure(pdb_path)

    binder_seq = extract_chain_sequence(structure, entry.chain_1)
    target_seq = extract_chain_sequence(structure, entry.chain_2)
    if not binder_seq or not target_seq:
        raise ValueError(f"Empty sequence for {entry}")

    case_dir = workdir / f"{entry.pdb_id}_{entry.chain_1}_{entry.chain_2}"
    case_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = case_dir / "input.fa"
    write_fasta_for_boltz(target_seq, binder_seq, fasta_path)

    boltz_out = case_dir / "boltz_output"
    run_boltz_predict(
        boltz_binary=boltz_binary,
        fasta_path=fasta_path,
        output_dir=boltz_out,
        extra_args=boltz_predict_extra_args,
        accelerator=accelerator,
        sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        recycling_steps=recycling_steps,
        use_msa_server=use_msa_server,
        cache_dir=boltz_cache_dir,
    )

    refold_cif, confidence_json, conf = choose_best_sample_and_metrics(boltz_out)

    design_iiptm = _safe_get(conf, "design_iiptm", _safe_get(conf, "design_to_target_iptm", np.nan))
    design_ptm = _safe_get(conf, "design_ptm", np.nan)
    min_pae = _safe_get(conf, "min_design_to_target_pae", np.nan)

    target_chain_refold, binder_chain_refold = detect_refold_chain_mapping(
        refold_cif=refold_cif,
        target_seq=target_seq,
        binder_seq=binder_seq,
    )
    delta_sasa, sasa_unbound, sasa_bound = compute_delta_sasa_refolded(
        refold_cif,
        target_chain=target_chain_refold,
        binder_chain=binder_chain_refold,
    )
    hbonds, salt = count_refolded_noncovalents(refold_cif, binder_chain=binder_chain_refold)

    return RefoldResult(
        entry=entry,
        pdb_path=pdb_path,
        refold_cif=refold_cif,
        confidence_json=confidence_json,
        design_iiptm=design_iiptm,
        design_ptm=design_ptm,
        min_design_to_target_pae=min_pae,
        neg_min_design_to_target_pae=-min_pae,
        delta_sasa_refolded=delta_sasa,
        design_sasa_unbound_refolded=sasa_unbound,
        design_sasa_bound_refolded=sasa_bound,
        plip_hbonds_refolded=hbonds,
        plip_saltbridge_refolded=salt,
    )


def main() -> None:
    args = parse_args()

    if args.install_requirements:
        maybe_install(["biotite", "hydride", "torch", "numpy", "pandas", "scipy", "biopython"])

    entries = read_entries(Path(args.input_csv))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(args.pdb_dir) if args.pdb_dir else workdir / "pdb_cache"
    boltz_cache_dir = Path(args.boltz_cache_dir) if args.boltz_cache_dir else workdir / "boltz_cache"
    boltz_binary_path = shutil.which(args.boltz_binary) if not Path(args.boltz_binary).exists() else args.boltz_binary
    if not args.validate_only and boltz_binary_path is None:
        raise RuntimeError(
            f"Could not find Boltz binary '{args.boltz_binary}' on PATH. "
            "Install Boltz or pass --boltz_binary /full/path/to/boltz. "
            "You can run with --validate_only to test PDB download and chain parsing without Boltz."
        )

    rows = []
    failures = []
    for e in entries:
        try:
            if args.validate_only:
                pdb_path = download_pdb(e.pdb_id, pdb_dir)
                structure = parse_structure(pdb_path)
                binder_seq = extract_chain_sequence(structure, e.chain_1)
                target_seq = extract_chain_sequence(structure, e.chain_2)
                rows.append(
                    {
                        "pdb_id": e.pdb_id,
                        "input_chain_1_binder": e.chain_1,
                        "input_chain_2_target": e.chain_2,
                        "source_pdb": str(pdb_path),
                        "binder_length": len(binder_seq),
                        "target_length": len(target_seq),
                    }
                )
            else:
                r = process_entry(
                    entry=e,
                    pdb_dir=pdb_dir,
                    workdir=workdir,
                    boltz_binary=boltz_binary_path,
                    boltz_predict_extra_args=args.boltz_predict_extra_args,
                    accelerator=args.accelerator,
                    sampling_steps=args.sampling_steps,
                    diffusion_samples=args.diffusion_samples,
                    recycling_steps=args.recycling_steps,
                    use_msa_server=args.use_msa_server,
                    boltz_cache_dir=boltz_cache_dir,
                )
                rows.append(
                    {
                        "pdb_id": e.pdb_id,
                        "input_chain_1_binder": e.chain_1,
                        "input_chain_2_target": e.chain_2,
                        "source_pdb": str(r.pdb_path),
                        "refold_cif": str(r.refold_cif),
                        "confidence_json": str(r.confidence_json),
                        "design_iiptm": r.design_iiptm,
                        "design_ptm": r.design_ptm,
                        "min_design_to_target_pae": r.min_design_to_target_pae,
                        "neg_min_design_to_target_pae": r.neg_min_design_to_target_pae,
                        "delta_sasa_refolded": r.delta_sasa_refolded,
                        "design_sasa_unbound_refolded": r.design_sasa_unbound_refolded,
                        "design_sasa_bound_refolded": r.design_sasa_bound_refolded,
                        "plip_hbonds_refolded": r.plip_hbonds_refolded,
                        "plip_saltbridge_refolded": r.plip_saltbridge_refolded,
                    }
                )
        except Exception as exc:
            failures.append({"pdb_id": e.pdb_id, "chain_1": e.chain_1, "chain_2": e.chain_2, "error": str(exc)})

    if not rows:
        raise RuntimeError(f"No successful entries. Failures: {failures}")

    ranked = pd.DataFrame(rows) if args.validate_only else rank_like_boltzgen(pd.DataFrame(rows))
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_csv, index=False)

    if failures:
        fail_path = Path(args.output_csv).with_suffix(".failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"Wrote failures to {fail_path}")

    print(f"Wrote ranked output to {args.output_csv}")


if __name__ == "__main__":
    main()
