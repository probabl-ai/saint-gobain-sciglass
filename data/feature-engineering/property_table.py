"""Step 2: Elemental property table S.

Cassar 2023 (arXiv:2303.15538 §2.2, first step):
Build one numeric table S for every in-scope element (77 elements).
55 properties collected from mendeleev and matminer:
  - 25 survivors from Table 1
  - 30 from Supplementary Material ("considered but not selected")
Any property with missing values on the 77 elements is dropped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from step01_atomic_fractions import ELEMENTS_77

logger = logging.getLogger(__name__)

# Table 1: 25 properties that survived selection
SURVIVORS_25: list[tuple[str, str, str]] = [
    ("vdw_radius_uff", "rW", "Van der Waals radius (UFF)"),
    ("atomic_radius_rahm", "rR", "Atomic radius (Rahm)"),
    ("atomic_volume", "Vat", "Atomic volume"),
    ("ElectronAffinity", "Eea", "Electron affinity"),
    ("GSbandgap", "Eg", "DFT bandgap, T = 0 K ground state"),
    ("GSenergy_pa", "Eat", "DFT energy per atom, T = 0 K ground state"),
    ("GSmagmom", "mm", "DFT magnetic moment, T = 0 K ground state"),
    ("GSestFCClatcnt", "FCClp", "FCC lattice parameter from OQMD DFT volume"),
    ("zeff", "Zeff", "Effective nuclear charge"),
    ("en_Sanderson", "χS", "Electronegativity, Sanderson"),
    ("en_Tardini_Organov", "χTO", "Electronegativity, Tardini-Oganov"),
    ("boiling_point", "Tb", "Boiling point"),
    ("FusionEnthalpy", "ΔHm", "Melting (fusion) enthalpy"),
    ("c6_gb", "C6", "C6 coefficient"),
    ("nvalence", "Nv", "Number of valence electrons"),
    ("num_oxistates", "Nox", "Number of oxidation states"),
    ("NUnfilled", "Nu", "Number of unfilled valence orbitals"),
    ("NsUnfilled", "Nu,s", "Unfilled s valence orbitals"),
    ("NpUnfilled", "Nu,p", "Unfilled p valence orbitals"),
    ("NdUnfilled", "Nu,d", "Unfilled d valence orbitals"),
    ("NfUnfilled", "Nu,f", "Unfilled f valence orbitals"),
    ("NsValence", "Nf,s", "Filled s valence orbitals"),
    ("NpValence", "Nf,p", "Filled p valence orbitals"),
    ("NdValence", "Nf,d", "Filled d valence orbitals"),
    ("NfValence", "Nf,f", "Filled f valence orbitals"),
]

# Supplementary: 30 properties considered but not selected
SUPPLEMENTARY_30: list[tuple[str, str, str]] = [
    ("atomic_number", "Z", "Atomic number"),
    ("atomic_weight", "Aw", "Atomic weight"),
    ("mass_number", "A", "Mass number of most abundant isotope"),
    ("max_ionenergy", "IEmax", "Maximum ionization energy"),
    ("electrons", "Nelec", "Number of electrons"),
    ("neutrons", "Nneut", "Number of neutrons"),
    ("protons", "Nprot", "Number of protons"),
    ("atomic_radius", "rSlater", "Atomic radius (Slater)"),
    ("covalent_radius_cordero", "rCordero", "Covalent radius (Cordero)"),
    ("covalent_radius_pyykko", "rPyykko", "Single-bond covalent radius (Pyykko)"),
    ("density", "rho295", "Density at 295 K"),
    ("heat_of_formation", "dHf", "Heat of formation"),
    ("melting_point", "Tm", "Melting point"),
    ("vdw_radius", "rVdW_CRC", "Van der Waals radius (CRC)"),
    ("dipole_polarizability", "alpha", "Dipole polarizability"),
    ("en_Allred_Rochow", "chi_AR", "Electronegativity (Allred-Rochow)"),
    ("en_Cottrell_Sutton", "chi_CS", "Electronegativity (Cottrell-Sutton)"),
    ("en_Gordy", "chi_Gordy", "Electronegativity (Gordy)"),
    ("en_Ghosh", "chi_Ghosh", "Electronegativity (Ghosh)"),
    ("en_Martynov_Batsanov", "chi_MB", "Electronegativity (Martynov-Batsanov)"),
    ("en_Nagle", "chi_Nagle", "Electronegativity (Nagle)"),
    ("FirstIonizationEnergy", "IE1", "Energy to remove first electron"),
    ("NValence", "Nv_matminer", "Number of valence electrons (matminer)"),
    ("GSestBCClatcnt", "BCClp", "BCC lattice parameter from OQMD DFT volume"),
    ("GSvolume_pa", "Vpa", "DFT volume per atom, T = 0 K ground state"),
    ("glawe_number", "N_Glawe", "Glawe number"),
    ("mendeleev_number", "N_Mendeleev", "Mendeleev number"),
    ("pettifor_number", "N_Pettifor", "Pettifor number"),
    ("vdw_radius_alvarez", "rVdW_Alvarez", "Van der Waals radius (Alvarez)"),
    ("vdw_radius_mm3", "rVdW_Allinger", "Van der Waals radius (Allinger/MM3)"),
]

ALL_PROPERTIES = SURVIVORS_25 + SUPPLEMENTARY_30
PROPERTY_NAMES = [name for name, _, _ in ALL_PROPERTIES]

_CACHED_TABLE: pd.DataFrame | None = None


def get_property_table(force_reload: bool = False) -> pd.DataFrame:
    """Return the cached (77, 55) numeric property table S for in-scope elements.

    The rows correspond exactly to ELEMENTS_77 in canonical order.
    Columns correspond to the 55 properties. Any property with a missing
    value on any of the 77 elements is dropped.
    """
    global _CACHED_TABLE
    if _CACHED_TABLE is not None and not force_reload:
        return _CACHED_TABLE

    fe_dir = Path(__file__).resolve().parent
    local_csv = fe_dir / "element_properties_55.csv"

    if local_csv.exists() and not force_reload:
        df = pd.read_csv(local_csv, index_col=0)
        _CACHED_TABLE = df.loc[ELEMENTS_77]
        return _CACHED_TABLE

    # Load from GlassPy bundled data (author compilation of mendeleev + matminer)
    import glasspy

    gp_data = (
        Path(glasspy.__file__).parent / "chemistry" / "data" / "chemical_properties.csv"
    )
    raw_df = pd.read_csv(gp_data, index_col=0)

    # Subset to the 77 elements and 55 properties
    sub_df = raw_df.loc[ELEMENTS_77, PROPERTY_NAMES].copy()

    # Drop any property with NaN on any of the 77 elements
    has_nan = sub_df.isna().any(axis=0)
    dropped = sub_df.columns[has_nan].tolist()
    if dropped:
        logger.warning(
            "Dropping %d properties due to missing values: %s", len(dropped), dropped
        )
        sub_df = sub_df.drop(columns=dropped)

    # Cache locally to disk
    sub_df.to_csv(local_csv)
    logger.info("Saved %s (shape %s)", local_csv, sub_df.shape)

    _CACHED_TABLE = sub_df
    return _CACHED_TABLE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    table = get_property_table(force_reload=True)
    print(
        f"Property table S: {table.shape[0]} elements x {table.shape[1]} props"
    )
    print(f"Total NaNs: {table.isna().sum().sum()}")
