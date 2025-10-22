import os
import sys
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from pyteomics import mgf
from joblib import Parallel, delayed
from src.ms_pred.common.chem_utils import standardize_adduct
from src.ms_pred.common.chem_utils import VALID_ELEMENTS

def contains_only_valid_elements(smiles, valid_elements=VALID_ELEMENTS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in valid_elements:
            return False
    return True

def process_spectrum(i, spec, spec_file_dir):
    params = spec.get("params", {})
    smiles = params.get("smiles")

    if not smiles or str(smiles).strip().lower() == "nan":
        return None
    if not contains_only_valid_elements(smiles):
        return None

    name = params.get("title", f"spec_{i}")
    spec_id = params.get("scans", f"{i}")
    try:
        base_filename = str(int(float(spec_id)))
    except Exception:
        base_filename = f"{i}"

    spec_file_path = os.path.join(spec_file_dir, f"{base_filename}.ms")

    try:
        mol = Chem.MolFromSmiles(smiles)
        inchikey = AllChem.MolToInchiKey(mol)
    except Exception:
        return None

    formula = params.get("formula", "")
    mol_mass = params.get("pepmass", [None])[0]
    raw_adduct = params.get("adduct", "M+H")

    if mol_mass is None:
        return None

    try:
        adduct = standardize_adduct(raw_adduct)
    except ValueError:
        return None

    try:
        with open(spec_file_path, "w") as f:
            f.write(f"#name {base_filename}\n")
            f.write(f"#parentmass {mol_mass:.4f}\n\n")
            f.write(f">{base_filename}\n")
            for mz, intensity in zip(spec["m/z array"], spec["intensity array"]):
                f.write(f"{mz:.4f} {intensity:.1f}\n")
    except Exception:
        return None

    return {
        "spec": base_filename,
        "smiles": smiles,
        "inchikey": inchikey,
        "mol_mass": mol_mass,
        "formula": formula,
        "name": name,
        "ionization": adduct
    }

def main():
    
    output_dir = "data/spec_datasets/nist20"
    input_mgf = os.path.join(output_dir, "nist_train_disjoint.mgf")

    spec_file_dir = os.path.join(output_dir, "spec_files")
    os.makedirs(spec_file_dir, exist_ok=True)

    all_spectra = list(mgf.read(input_mgf))
    print(f"Loaded {len(all_spectra)} spectra from {input_mgf}")
    print(f"Writing spec files to {spec_file_dir} and labels to {os.path.join(output_dir, 'labels.tsv')}")

    results = Parallel(n_jobs=12)(
        delayed(process_spectrum)(i, spec, spec_file_dir)
        for i, spec in tqdm(enumerate(all_spectra), total=len(all_spectra), desc="Processing spectra")
    )

    labels = [r for r in results if r is not None]
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(os.path.join(output_dir, "labels.tsv"), sep="\t", index=False)

if __name__ == "__main__":
    main()
