import datetime
import json
import math
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
import seaborn as sns
from mzqc import MZQCFile as qc
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


def read_mzqc(filename: str) -> Tuple[pd.DataFrame, dict]:
    """
    Read an mzQC file and return a DataFrame with the metrics.

    - Single values are reported in a single column with the format
      <accession>.
    - n-tuple values are reported in n columns with the format
      <accession>_<index>.
    - Table values are reported in n columns with the format
      <accession>_<key>_<index>.


    Parameters
    ----------
    filename : str
        Path to the mzQC file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the metrics.
    dict
        Dictionary mapping accession to metric name.
    """
    with open(filename) as f_in:
        mzqc_data = qc.JsonSerialisable.from_json(f_in)

    metrics = dict()
    accession_to_name = dict()
    for run in mzqc_data.runQualities:
        run_label = run.metadata.label
        metrics[run_label] = dict()
        for metric in run.qualityMetrics:
            if isinstance(metric.value, (int, float)):
                metrics[run_label][metric.accession] = metric.value
                accession_to_name[metric.accession] = metric.name
            elif isinstance(metric.value, list):
                for i, value in enumerate(metric.value):
                    metrics[run_label][f"{metric.accession}_{i}"] = value
                    accession_to_name[f"{metric.accession}_{i}"] = f"{metric.name}"

    return pd.DataFrame(metrics).T, accession_to_name


def preprocess(
    data: pd.DataFrame, min_variance: float, min_corr: float, scaling_mode: str
) -> pd.DataFrame:
    """
    Preprocess the data by removing low-variance and correlated metrics,
    and scaling the values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with the metrics.
    min_variance : float
        Minimum variance to keep the metric.
    min_corr : float
        Minimum correlation to remove a metric.
    scaling_mode : str
        Scaling mode, either "standard" or "robust".

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame.
    """
    # Remove low-variance metrics.
    selector = VarianceThreshold(min_variance).fit(data)
    data = data[data.columns[selector.get_support(indices=True)]]

    # Remove correlated metrics.
    mask = np.ones(len(data.columns), dtype=bool)
    corr = data.corr().abs()
    for i in range(len(data.columns)):
        if not mask[i]:
            continue
        for j in range(i + 1, len(data.columns)):
            if corr.iloc[i, j] > min_corr:
                mask[j] = False
    data = data[data.columns[mask]]

    # Scale the values.
    scaler = RobustScaler() if scaling_mode == "robust" else StandardScaler()
    data = pd.DataFrame(
        scaler.fit_transform(data), index=data.index, columns=data.columns
    )

    return data


def detect_outliers_loop(
    data: pd.DataFrame, n_neighbors: int, lmbd: int = 3
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Detect outliers using the LoOP algorithm.

    Additionally, compute the per-feature attributions for each sample.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with the metrics.
    n_neighbors : int
        The number of neighbors to consider.
    lmbd : int
        The PLOF normalization factor.

    Returns
    -------
    np.ndarray
        Array with the LoOP scores.
    """
    # Compute the nearest neighbors for all points.
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn_dists, knn_ind = knn.fit(data).kneighbors()
    # Throw away the self-match.
    knn_dists, knn_ind = knn_dists[:, 1:], knn_ind[:, 1:]

    # Compute the probabilistic set distances (pdist).
    pdists = np.sqrt(np.sum(knn_dists**2, axis=1) / n_neighbors)

    # Compute Probabilistic Outlier Factor (PLOF) values.
    plofs = pdists / pdists[knn_ind].mean(axis=1) - 1
    nplof = lmbd * np.sqrt(np.mean(plofs**2))

    # Compute the LoOP scores.
    loop = np.clip(scipy.special.erf(plofs / (nplof * np.sqrt(2))), 0, 1)
    loop = pd.Series(loop, index=data.index)

    # Determine the relevant features for each outlier.
    attributions = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
    for i in range(data.shape[0]):
        # The outlier contribution of a sample is determined by the
        # per-feature squared differences to the average neighbour
        # features.
        diffs = (data.iloc[[i]].values - data.iloc[knn_ind[i]].values) ** 2
        per_feat = diffs.mean(axis=0)
        attributions.iloc[i] = per_feat / per_feat.sum()

    return loop, attributions


def export_mzqc(
    outlier_scores: pd.Series, output_filename: str, input_filename: str
) -> None:
    """
    Export the outlier scores to a setQuality mzQC file.

    Parameters
    ----------
    outlier_scores : pd.Series
        Series with the outlier scores.
    output_filename : str
        Path to the output mzQC file.
    input_filename : str
        Path to the original input mzQC file.
    """
    cv_url = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
    cv_version = "4.1.199"
    cv = qc.ControlledVocabulary(
        name="Proteomics Standards Initiative Mass Spectrometry Ontology",
        uri=cv_url,
        version=cv_version,
    )

    input_filename = pathlib.Path(input_filename)
    metadata = qc.MetaDataParameters(
        label=pathlib.Path(output_filename).stem,
        inputFiles=[
            qc.InputFile(
                name=input_filename.stem,
                location=input_filename.resolve().as_uri(),
                fileFormat=qc.CvParameter(
                    accession="MS:1003160",
                    name="mzQC format",
                ),
            )
        ],
        analysisSoftware=[
            qc.AnalysisSoftware(
                accession="MS:1000799",
                name="custom unreleased software tool",
                value="quality control LoOP outlier detection",
                version="2016-03-14",
                uri="https://pubs.acs.org/doi/full/10.1021/acs.jproteome.6b00028",
            )
        ],
    )
    set_quality = qc.SetQuality(
        metadata=metadata,
        qualityMetrics=[
            qc.QualityMetric(
                accession="MS:4000201",
                name="run outlier score LoOP",
                value={
                    # CV: mzQC input reference.
                    "MS:4000086": outlier_scores.index.tolist(),
                    # CV: algorithmical threshold.
                    "NCIT:C54154": outlier_scores.values.round(5).tolist(),
                },
            )
        ],
    )

    mzqc_file = qc.MzQcFile(
        creationDate=datetime.datetime.now().isoformat(timespec="seconds"),
        version="1.0.0",
        setQualities=[set_quality],
        controlledVocabularies=[cv],
    )

    with open(output_filename, "w") as f_out:
        f_out.write(
            json.dumps(
                json.loads(
                    qc.JsonSerialisable.to_json(mzqc_file, readability=0)
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    # Outlier detection hyperparameters.
    min_var = 0.01
    min_corr = 0.9
    scaling = "standard"
    n_neighbors = 20
    lmbd = 3

    filename_in = "../data/Mtb-120-outlier-metrics.mzQC"
    filename_out = "../data/Mtb-120-outlier-scores.mzQC"

    # Read the runQuality metrics from the mzQC file.
    metrics, accession_to_name = read_mzqc(filename_in)

    # Preprocess the data and compute outlier scores.
    metrics = preprocess(metrics, min_var, min_corr, scaling)
    outlier_scores, attributions = detect_outliers_loop(
        metrics, n_neighbors, lmbd
    )

    # Plot excessive outliers.
    outlier_threshold = 0.66
    outlier_scores = outlier_scores.sort_values(ascending=False)
    outlier_idx = outlier_scores.index[outlier_scores > outlier_threshold]
    nrows = math.ceil(len(outlier_idx) / 2)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=2, sharey=True, figsize=(9, 3 * nrows)
    )
    axes = axes.reshape(-1, 2)

    for i, ax in zip(outlier_idx, axes.flatten()):
        attr_metrics = attributions.loc[i].nlargest(5)
        attr_metrics.index = attr_metrics.index.map(accession_to_name)
        attr_metrics.plot.bar(ax=ax)

        ax.set_ylim(0, 0.5)

        ax.set_title(f"{i} (LoOP score = {outlier_scores.at[i]:.2f})")

        sns.despine(ax=ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("Feature attribution ratio")

    plt.tight_layout()
    plt.savefig(
        "attributions.png", dpi=300, bbox_inches="tight", transparent=True
    )
    plt.close()
