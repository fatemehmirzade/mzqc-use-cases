import argparse
import datetime
import json
import os
import pathlib
import urllib.parse
import urllib.request
from dataclasses import dataclass

import fastobo
import pandas as pd
from mzqc import MZQCFile as qc


@dataclass
class CvTerm:
    accession: str
    name: str = None
    description: str = None
    value_type: str = None
    unit: str = None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert QuaMeter output to mzQC format."
    )
    parser.add_argument(
        "--input_file", required=True, help="Path to the QuaMeter TSV file."
    )
    parser.add_argument(
        "--location_prefix",
        required=True,
        help="ProteomeXchange URL or path to the location of the raw files.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Path to save the mzQC output. Defaults to <input_file_basename>.mzQC.",
    )
    return parser.parse_args()


def load_cv_terms():
    cv_url = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
    cv_terms = {}
    with urllib.request.urlopen(cv_url) as response:
        cv = fastobo.load(response)
        cv_version = next(
            (
                clause.raw_value()
                for clause in cv.header
                if isinstance(clause, fastobo.header.DataVersionClause)
            ),
            "unknown",
        )
        for frame in cv:
            cv_term = CvTerm(str(frame.id))
            for clause in frame:
                if isinstance(clause, fastobo.term.NameClause):
                    cv_term.name = clause.name.strip()
                elif isinstance(clause, fastobo.term.DefClause):
                    cv_term.description = clause.definition.strip()
                elif isinstance(clause, fastobo.term.IsAClause):
                    cv_term.value_type = clause.raw_value()
                elif isinstance(
                    clause, fastobo.term.RelationshipClause
                ) and clause.raw_value().startswith("has_units"):
                    cv_term.unit = clause.raw_value().rsplit(" ", 1)[1]
            cv_terms[cv_term.accession] = cv_term
    return cv_version, cv_terms


def parse_quameter_file(input_file):
    return pd.read_csv(
        input_file,
        sep="\t",
        index_col="Filename",
        parse_dates=["StartTimeStamp"],
    )


def create_run_quality(metrics, cv_terms, location_prefix, filename):
    runs = []
    for run_filename, row in metrics.iterrows():
        # Run metadata: input files and software.
        accession = {
            ".raw": "MS:1000563",
            ".mzxml": "MS:1000566",
            ".mzml": "MS:1000584",
        }.get(os.path.splitext(run_filename)[1].lower(), "MS:1000560")
        # Default: unknown file format.

        peak_file = qc.InputFile(
            name=os.path.splitext(run_filename)[0],
            location=urllib.parse.urljoin(location_prefix, run_filename),
            fileFormat=qc.CvParameter(
                accession,
                cv_terms[accession].name,
                cv_terms[accession].description,
            ),
            fileProperties=[
                # Completion time.
                qc.CvParameter(
                    accession="MS:1000747",
                    name=cv_terms["MS:1000747"].name,
                    description=cv_terms["MS:1000747"].description,
                    value=row["StartTimeStamp"]
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z"),
                )
            ],
        )

        quameter_file = qc.InputFile(
            name=os.path.splitext(filename)[0],
            location=pathlib.Path(filename).resolve().as_uri(),
            # Tab delimited text format.
            fileFormat=qc.CvParameter(
                accession="MS:1000914",
                name=cv_terms["MS:1000914"].name,
                description=cv_terms["MS:1000914"].description,
            ),
        )

        # QuaMeter IDFree.
        software = qc.AnalysisSoftware(
            accession="MS:1003164",
            name=cv_terms["MS:1003164"].name,
            description=cv_terms["MS:1003164"].description,
            version="1.1.21142",
            uri="https://proteowizard.sourceforge.io/",
        )

        metadata = qc.MetaDataParameters(
            label=os.path.splitext(run_filename)[0],
            inputFiles=[peak_file, quameter_file],
            analysisSoftware=[software],
        )

        # Convert the individual metrics to their CV-supported versions.
        mapping = {
            "MS:4000050": ["XIC-WideFrac"],
            "MS:4000051": ["XIC-FWHM-Q1", "XIC-FWHM-Q2", "XIC-FWHM-Q3"],
            "MS:4000052": ["XIC-Height-Q2", "XIC-Height-Q3", "XIC-Height-Q4"],
            "MS:4000053": ["RT-Duration"],
            "MS:4000054": ["RT-TIC-Q1", "RT-TIC-Q2", "RT-TIC-Q3", "RT-TIC-Q4"],
            "MS:4000055": ["RT-MS-Q1", "RT-MS-Q2", "RT-MS-Q3", "RT-MS-Q4"],
            "MS:4000056": [
                "RT-MSMS-Q1",
                "RT-MSMS-Q2",
                "RT-MSMS-Q3",
                "RT-MSMS-Q4",
            ],
            "MS:4000057": [
                "MS1-TIC-Change-Q2",
                "MS1-TIC-Change-Q3",
                "MS1-TIC-Change-Q4",
            ],
            "MS:4000058": ["MS1-TIC-Q2", "MS1-TIC-Q3", "MS1-TIC-Q4"],
            "MS:4000059": ["MS1-Count"],
            "MS:4000060": ["MS2-Count"],
            "MS:4000061": [
                "MS1-Density-Q1",
                "MS1-Density-Q2",
                "MS1-Density-Q3",
            ],
            "MS:4000062": [
                "MS2-Density-Q1",
                "MS2-Density-Q2",
                "MS2-Density-Q3",
            ],
            "MS:4000063": [
                "MS2-PrecZ-1",
                "MS2-PrecZ-2",
                "MS2-PrecZ-3",
                "MS2-PrecZ-4",
                "MS2-PrecZ-5",
                "MS2-PrecZ-more",
            ],
            "MS:4000064": ["MS2-PrecZ-likely-1", "MS2-PrecZ-likely-multi"],
            "MS:4000065": ["MS1-Freq-Max"],
            "MS:4000066": ["MS2-Freq-Max"],
        }
        run_metrics = []
        for accession, col_names in mapping.items():
            cv_term = cv_terms[accession]

            if cv_term.value_type == "MS:4000003":  # Single value
                value = row[col_names[0]]
            elif cv_term.value_type == "MS:4000004":  # n-tuple
                value = [row[col] for col in col_names]
            elif cv_term.value_type == "MS:4000005":  # Table
                value = {
                    # Charge state
                    "MS:1000041": list(range(1, len(col_names) + 1)),
                    # Fraction
                    "UO:0000191": [row[col] for col in col_names],
                }
            else:
                raise ValueError(
                    f"Unsupported value type: {cv_term.value_type}"
                )

            unit = (
                qc.CvParameter(
                    accession=cv_term.unit, name=cv_terms[cv_term.unit].name
                )
                if cv_term.unit
                else None
            )

            run_metrics.append(
                qc.QualityMetric(
                    accession=accession,
                    name=cv_term.name,
                    description=cv_term.description,
                    value=value,
                    unit=unit,
                )
            )
        runs.append(
            qc.RunQuality(metadata=metadata, qualityMetrics=run_metrics)
        )
    return runs


def export_mzqc(runs, cv_version, output_file):
    cv_url = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
    cvs = [
        qc.ControlledVocabulary(
            name="Proteomics Standards Initiative Mass Spectrometry Ontology",
            uri=cv_url,
            version=cv_version,
        ),
    ]
    mzqc = qc.MzQcFile(
        creationDate=datetime.datetime.now().isoformat(timespec="seconds"),
        version="1.0.0",
        runQualities=runs,
        controlledVocabularies=cvs,
    )
    with open(output_file, "w") as f_out:
        f_out.write(
            json.dumps(
                json.loads(qc.JsonSerialisable.to_json(mzqc, readability=0)),
                indent=2,
            )
        )


def main():
    args = parse_arguments()
    if not args.output_file:
        args.output_file = (
            f"{os.path.splitext(os.path.basename(args.input_file))[0]}.mzQC"
        )
    cv_version, cv_terms = load_cv_terms()
    metrics = parse_quameter_file(args.input_file)
    runs = create_run_quality(
        metrics, cv_terms, args.location_prefix, args.input_file
    )
    export_mzqc(runs, cv_version, args.output_file)


if __name__ == "__main__":
    main()
