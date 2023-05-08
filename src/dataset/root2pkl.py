import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mplhep as hep
import joblib
from pyparsing import Union

from utils.io import check_inputpath, check_outputpath, logging_setup


def flatten_TLorentzVector(TLorentzVector_ak, branch_key):
    assert "TLorentzVector" in TLorentzVector_ak.typestr
    assert ["fP", "fE"] == TLorentzVector_ak.fields

    fP = ak.unzip(TLorentzVector_ak)[0]
    fE = ak.unzip(TLorentzVector_ak)[1]
    fX, fY, fZ = ak.unzip(fP)
    df = pd.DataFrame(
        {
            f"{branch_key}.fX": fX,
            f"{branch_key}.fY": fY,
            f"{branch_key}.fZ": fZ,
            f"{branch_key}.fE": fE,
        }
    )

    return df


def root2pkl(
    input_rootfile_path: Union[Path, str],
    output_folder_path: Union[Path, str],
    output_file_name: str = None,
):
    """Convert ROOT file to pickle file, TTree -> pandas.DataFrame."""
    import logging

    input_rootfile_path = check_inputpath(input_rootfile_path)
    output_folder_path = check_outputpath(output_folder_path)
    if output_file_name is None:
        output_file_name = input_rootfile_path.stem
    output_file_path = output_folder_path / f"{output_file_name}.pkl"

    logging_setup(
        verbosity=3,
        if_write_log=True,
        output_path=output_folder_path,
        filename=f"root2pkl_{input_rootfile_path.name}",
    )

    logging.info(
        f"Converting {input_rootfile_path.resolve()} to \n {output_file_path.resolve()} ..."
    )
    logging.info(f"Reading {input_rootfile_path.stem} ...")

    try:
        tree = uproot.open(input_rootfile_path)["NOMINAL"]
        branch_types = tree.typenames()
    except Exception as e:
        logging.error(f"Failed to open {input_rootfile_path.resolve()}. ")
        raise e

    simple_branch_keys = []
    TLorentzVector_branch_keys = []
    std_vector_branch_keys = []

    for branch_key, branch_type in branch_types.items():
        if (
            branch_type.startswith("int")
            or branch_type.startswith("float")
            or branch_type.startswith("bool")
            or branch_type.startswith("uint")
        ):
            simple_branch_keys.append(branch_key)
            continue

        if branch_type == "TLorentzVector":
            TLorentzVector_branch_keys.append(branch_key)
            continue

        if branch_type.startswith("std::vector"):
            std_vector_branch_keys.append(branch_key)
            continue

    simple_branch_df = tree.arrays(
        filter_name=simple_branch_keys, library="pd"
    )
    TLorentzVector_branchs_ak = tree.arrays(
        filter_name=TLorentzVector_branch_keys, library="ak"
    )
    std_vector_branchs_ak = tree.arrays(
        filter_name=std_vector_branch_keys, library="ak"
    )  # jets related branches, variable length

    TLorentzVector_branchs_df = pd.DataFrame()
    for TLorentzVector_branch_key in TLorentzVector_branch_keys:
        TLorentzVector_branch_ak = TLorentzVector_branchs_ak[
            TLorentzVector_branch_key
        ]
        TLorentzVector_df = flatten_TLorentzVector(
            branch_key=TLorentzVector_branch_key,
            TLorentzVector_ak=TLorentzVector_branch_ak,
        )
        TLorentzVector_branchs_df = pd.concat(
            [TLorentzVector_branchs_df, TLorentzVector_df], axis=1
        )

    ### std_vector_branchs_keys contain both simple types and TLorentzVector types as std::vector<type>
    ### trying to flatten 2 jets... but it depends on the number of jets in the event
    ### one could add some cuts.
    std_vector_branchs_df = pd.DataFrame()
    std_vector_branch_types = [
        branch_types[branch_key] for branch_key in std_vector_branch_keys
    ]
    std_vector_TLorentzVector_branch_keys = []
    while "std::vector<TLorentzVector>" in std_vector_branch_types:
        idx = std_vector_branch_types.index("std::vector<TLorentzVector>")
        std_vector_branch_types.pop(idx)
        std_vector_TLorentzVector_branch_keys.append(
            std_vector_branch_keys.pop(idx)
        )

    ### std_vector_branchs_keys only contain simple types now
    std_vector_simpletype_branchs_keys = std_vector_branch_keys

    n_jets_kept = 2
    while n_jets_kept > 0:
        try:
            std_vector_branchs_ak[:, n_jets_kept - 1]
            break
        except IndexError:
            logging.error(f"Keeping {n_jets_kept} jets not successful.")
            n_jets_kept -= 1

    logging.info(f"Keeping {n_jets_kept} jets")
    jets_kept = std_vector_branchs_ak[:, :n_jets_kept]

    for jet_idx in range(n_jets_kept):
        logging.debug(f"Processing jet {jet_idx} ...")
        jet_simpletype_df = ak.to_dataframe(
            jets_kept[std_vector_simpletype_branchs_keys, :, jet_idx]
        )
        jet_simpletype_df = jet_simpletype_df.add_prefix(f"jet{jet_idx}.")

        for (
            std_vector_TLorentzVector_branch_key
        ) in std_vector_TLorentzVector_branch_keys:
            jet_simpletype_df = pd.concat(
                [
                    jet_simpletype_df,
                    flatten_TLorentzVector(
                        jets_kept[
                            std_vector_TLorentzVector_branch_key, :, jet_idx
                        ],
                        f"jet{jet_idx}",
                    ),
                ],
                axis=1,
            )

        std_vector_branchs_df = pd.concat(
            [std_vector_branchs_df, jet_simpletype_df], axis=1
        )

    root_df = pd.concat(
        [simple_branch_df, TLorentzVector_branchs_df, std_vector_branchs_df],
        axis=1,
    )
    joblib.dump(root_df, output_file_path)
    logging.info(f"Saved {output_file_path.resolve()}. Done. ")
    return root_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ROOT file to pickle file, TTree -> pandas.DataFrame. "
    )
    parser.add_argument(
        "input-rootfile-path",
        type=str,
        required=True,
        help="Path to the input ROOT file. ",
    )
    parser.add_argument(
        "output-folder-path",
        type=str,
        required=True,
        help="Path to the output folder. ",
    )
    parser.add_argument(
        "--output-file-name",
        type=str,
        default=None,
        help="Name of the output file. ",
    )
    args = parser.parse_args()

    root2pkl(
        input_rootfile_path=args.input_rootfile_path,
        output_folder_path=args.output_folder_path,
        output_file_name=args.output_file_name,
    )
