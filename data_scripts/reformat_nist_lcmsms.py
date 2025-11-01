#!/usr/bin/env python3
"""

- Reads a JSON file (array) with keys like:
  spectrum_id, peaks_json, Compound_Name, Adduct, Precursor_MZ, Smiles,
  Ion_Mode, Instrument, collision_energy, ...

- Produces:
  - spec_files.hdf5  (HDF5 of .ms text blocks via common.HDF5Dataset)
  - mgf_files/nist_all.mgf  (merged peaks with m/z rounding & intensity sum)
  - labels.tsv

Requires: rdkit, numpy, pandas, tqdm, pathos, multiprocess, ms_pred.common
"""

from pathlib import Path
import re
import json
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import ms_pred.common as common

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm
import multiprocess.context as ctx
ctx._force_start_method('spawn')
from pathos import multiprocessing as mp

# ========================= CONFIG — EDIT ME =========================
INPUT_JSON   = Path("data/spec_datasets/nist20/filtered_library.json")
TARG_DIR     = Path("data/spec_datasets/nist20/")
WORKERS      = 16
INSTR_POLICY = "hcd-only"
ROUND_PREC   = 4
# ===================================================================

COLLISION_NUM_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
VALID_ELS = {
    "C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B",
    "Se", "Fe", "Co", "As", "Na", "K"
}
ION_MAP = {
    '[M+H-H2O]+': '[M-H2O+H]+',
    '[M+NH4]+':   '[M+H3N+H]+',
    '[M+H-2H2O]+':'[M-H4O2+H]+',
}
RELAXED_INSTRUMENTS = {"Q-TOF", "HCD", "CID", "QTOF", "QTOF/HCD", "QQQ"}


def get_els(formula: str):
    return {i[0] for i in re.findall(r"([A-Z][a-z]*)([0-9]*)", formula)}


def chunked_parallel(input_list, function, chunks=100, max_cpu=16):
    if len(input_list) == 0:
        return []
    cpus = min(mp.cpu_count(), max_cpu)
    pool = mp.Pool(processes=cpus)

    def batch_func(lst):
        return [function(i) for i in lst]

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step = max(1, list_len // num_chunks)
    chunks_ = [input_list[i:i+step] for i in range(0, list_len, step)]
    outputs = list(tqdm(pool.imap(batch_func, chunks_), total=len(chunks_)))
    return [y for x in outputs for y in x]


def build_mgf_str(
    meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]],
    merge_charges=True,
    parent_mass_keys=("PEPMASS", "parentmass", "PRECURSOR_MZ"),
    precision=ROUND_PREC,
) -> str:
    entries = []
    for meta, spec in tqdm(meta_spec_list, desc="Writing MGF"):
        rows = ["BEGIN IONS"]

        # Precursor mass
        for k in parent_mass_keys:
            if k in meta:
                try:
                    pep_mass = float(meta.get(k, -100))
                    rows.append(f"PEPMASS={pep_mass}")
                    break
                except Exception:
                    pass

        # Metadata as KEY=VALUE with underscores
        for k, v in meta.items():
            rows.append(f"{k.upper().replace(' ', '_')}={v}")

        if merge_charges:
            spec_ar = np.vstack([i[1] for i in spec]) if len(spec) else np.empty((0, 2))
            mz_to_int = {}
            for mz, inten in spec_ar:
                mz = float(np.round(mz, precision))
                mz_to_int[mz] = mz_to_int.get(mz, 0.0) + float(inten)
            merged = np.array(sorted([[m, v] for m, v in mz_to_int.items()]), dtype=float)
        else:
            raise NotImplementedError()

        rows.extend([f"{m} {i}" for m, i in merged])
        rows.append("END IONS")
        entries.append("\n".join(rows))

    return "\n\n".join(entries)


def uncharged_formula(mol_or_smiles: str, mol_type="smiles") -> str:
    if mol_type == "mol":
        mol = mol_or_smiles
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            return None
    else:
        raise ValueError("mol_type must be 'mol' or 'smiles'")
    chem_formula = CalcMolFormula(mol)
    m = re.findall(r"^([^\+,^\-]*)", chem_formula)
    return m[0] if m else chem_formula


def normalize_instrument(instr: str) -> str:
    if not instr:
        return "UNKNOWN"
    s = instr.strip().upper().replace(" ", "")
    if s in {"QTOF", "Q-TOF", "Q_TOF"}:
        return "Q-TOF"
    return s


def parse_peaks(peaks_raw: Any) -> np.ndarray:
    if isinstance(peaks_raw, str):
        peaks_raw = json.loads(peaks_raw)
    arr = np.array(peaks_raw, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("peaks_json must be pairs [mz, intensity]")
    return arr[:, :2]


def fails_filter(entry: Dict[str, Any],
                 valid_adduct=None,
                 max_mass: float = 1500.0,
                 instrument_policy: str = INSTR_POLICY) -> bool:
    if valid_adduct is None:
        valid_adduct = list(common.ion2mass.keys())

    adduct = entry.get('PRECURSOR TYPE', '')
    if adduct in ION_MAP:
        adduct = ION_MAP[adduct]
    if adduct not in valid_adduct:
        return True

    if 'EXACT MASS' not in entry:
        return True
    try:
        if float(entry['EXACT MASS']) > max_mass:
            return True
    except Exception:
        return True

    inst = entry.get('INSTRUMENT TYPE', 'UNKNOWN').upper()
    if instrument_policy == "hcd-only":
        if inst != "HCD":
            return True
    else:  # "all"
        if inst != "UNKNOWN" and inst not in RELAXED_INSTRUMENTS:
            return True

    form = entry.get('FORMULA')
    if not form:
        return True
    form_els = get_els(form)
    if len(form_els.intersection(VALID_ELS)) != len(form_els):
        return True

    return False


def process_json_entry(entry: Dict[str, Any],
                       instrument_policy: str = INSTR_POLICY) -> Dict[str, Any]:
    try:
        out = {}
        out["spec_id"] = entry.get("spectrum_id", entry.get("spec_id", "unknown"))
        out["SYNONYMS"] = entry.get("Compound_Name", "")

        adduct_raw = entry.get("Adduct", "")
        if adduct_raw and not adduct_raw.startswith("["):
            adduct = f"[{adduct_raw}]+"
        else:
            adduct = adduct_raw
        adduct = ION_MAP.get(adduct, adduct)
        out["PRECURSOR TYPE"] = adduct

        out["PRECURSOR M/Z"] = float(entry["Precursor_MZ"])

        out["INSTRUMENT TYPE"] = normalize_instrument(entry.get("Instrument", ""))
        out["SPECTRUM TYPE"] = "MS2"

        ce = entry.get("collision_energy", "")
        out["COLLISION ENERGY"] = str(ce)

        peaks = parse_peaks(entry.get("peaks_json", []))
        out["Peaks"] = peaks

        smi = entry.get("Smiles", "")
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            return {}
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = Chem.MolFromSmiles(smiles)
        out["smiles"] = smiles
        out["INCHIKEY"] = Chem.MolToInchiKey(mol)
        out["FORMULA"] = uncharged_formula(smiles, mol_type="smiles")
        out["EXACT MASS"] = float(entry.get("Precursor_MZ"))

        if fails_filter(out, instrument_policy=instrument_policy):
            return {}

        return out

    except Exception as e:
        print(f"Skipping entry due to error: {e}")
        return {}


def load_json_entries(json_path: Path,
                      instrument_policy: str = INSTR_POLICY) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be an array of records.")
    outputs = []
    for ent in tqdm(data, desc="Parsing JSON"):
        out = process_json_entry(ent, instrument_policy=instrument_policy)
        if out:
            outputs.append(out)
    return outputs


def merge_data(collision_dict: dict):
    base_dict = None
    out_peaks = {}
    num_peaks = 0
    energies = []
    for energy, sub_dict in collision_dict.items():
        if base_dict is None:
            base_dict = sub_dict
        if energy in out_peaks:
            raise ValueError(f"Duplicate energy {energy}")
        out_peaks[energy] = np.array(sub_dict["Peaks"], dtype=float)
        energies.append(energy)
        num_peaks += len(out_peaks[energy])

    base_dict["Peaks"] = out_peaks
    base_dict["COLLISION ENERGY"] = energies
    base_dict["NUM PEAKS"] = num_peaks

    peak_list = list(base_dict.pop("Peaks").items())
    info_dict = base_dict
    return (info_dict, peak_list)


def dump_fn(entry: tuple) -> (dict, dict):
    entry, peaks = entry
    output_name = entry["spec_id"]
    common_name = entry.get("SYNONYMS", "")
    formula = entry["FORMULA"]
    ionization = entry["PRECURSOR TYPE"]
    parent_mass = entry["PRECURSOR M/Z"]
    instrument = entry.get("INSTRUMENT TYPE", "UNKNOWN")

    out_entry = {
        "dataset": "nist2020",
        "spec": output_name,
        "name": common_name,
        "formula": formula,
        "ionization": ionization,
        "smiles": entry["smiles"],
        "inchikey": entry["INCHIKEY"],
        "precursor": parent_mass,
        "collision_energies": [k for k, _ in peaks],
        "instrument": instrument,   
    }

    exclude_comments = {"Peaks"}
    header_str = "\n".join([
        f">compound {common_name}",
        f">formula {formula}",
        f">ionization {ionization}",
        f">parentmass {parent_mass}",
    ])
    comment_str = "\n".join([f"#{k} {v}" for k, v in entry.items() if k not in exclude_comments])

    peak_blocks = []
    for k, v in peaks:
        block = [f">collision {k}"]
        block.extend([f"{row[0]} {row[1]}" for row in v])
        peak_blocks.append("\n".join(block))

    out_str = header_str + "\n" + comment_str + "\n\n" + "\n\n".join(peak_blocks)
    return out_entry, {f"{output_name}.ms": out_str}


def main():
    # Prepare dirs & targets
    target_directory = TARG_DIR
    target_directory.mkdir(exist_ok=True, parents=True)
    target_ms = target_directory / "spec_files.hdf5"
    target_mgf = target_directory / "mgf_files"
    target_labels = target_directory / "labels.tsv"
    target_mgf.mkdir(exist_ok=True, parents=True)

    # Load & preprocess JSON
    output_dicts = load_json_entries(INPUT_JSON, instrument_policy=INSTR_POLICY)

    # Build {inchikey → adduct → instrument → collision_energy → dict}
    parsed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    print("Shuffling dict before merge")
    for outd in tqdm(output_dicts):
        inchikey = outd["INCHIKEY"]
        adduct = outd["PRECURSOR TYPE"]
        instrument = outd.get("INSTRUMENT TYPE", "UNKNOWN")
        ce_str = outd.get("COLLISION ENERGY", "")
        nums = COLLISION_NUM_RE.findall(str(ce_str))
        if not nums:
            continue
        ce_val = nums[-1]  # take the last numeric token (e.g., "25 eV" -> "25")
        parsed[inchikey][adduct][instrument][ce_val] = outd

    # Merge entries per (inchikey, adduct, instrument)
    print("Merging dicts")
    merged_entries = []
    for _inchikey, adduct_dict in tqdm(parsed.items()):
        for _adduct, instrument_dict in adduct_dict.items():
            for _instrument, collision_dict in instrument_dict.items():
                merged_entries.append(merge_data(collision_dict))

    # Dump outputs
    print("Export to file")
    if WORKERS <= 1:
        output_tuples = [dump_fn(i) for i in merged_entries]
    else:
        output_tuples = chunked_parallel(merged_entries, dump_fn, chunks=1000, max_cpu=WORKERS)

    output_entries = []
    ms_entries = {}
    for tup in output_tuples:
        output_entries.append(tup[0])
        ms_entries.update(tup[1])

    # HDF5 of .ms files
    h5 = common.HDF5Dataset(target_ms, 'w')
    h5.write_dict(ms_entries)
    h5.close()

    # One merged MGF across all energies
    mgf_out = build_mgf_str(merged_entries, precision=ROUND_PREC)
    (target_mgf / "nist_all.mgf").write_text(mgf_out)

    # Labels
    df = pd.DataFrame(output_entries)
    df['ionization'] = [ION_MAP.get(i, i) for i in df['ionization'].values]
    df.to_csv(target_labels, sep="\t", index=False)

    print("\nDone.")
    print(f"HDF5 : {target_ms.resolve()}")
    print(f"MGF  : {(target_mgf / 'nist_all.mgf').resolve()}")
    print(f"TSV  : {target_labels.resolve()}")


if __name__ == "__main__":
    main()
