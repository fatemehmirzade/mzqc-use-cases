import functools

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mzqc import MZQCFile as qc


plt.style.use(["seaborn-v0_8-white", "seaborn-v0_8-paper"])
sns.set_palette(["#9e0059", "#6da7de", "#ee266d", "#dee000", "#eb861e"])
sns.set_context("paper")


def read_mzqc(filename: str) -> pd.DataFrame:
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
    """
    with open(filename) as f_in:
        mzqc_data = qc.JsonSerialisable.from_json(f_in)

    metrics = dict()
    for set in mzqc_data.setQualities:
        set_label = set.metadata.label
        metrics[set_label] = dict()
        for metric in set.qualityMetrics:
            if isinstance(metric.value, (int, float)):
                metrics[set_label][metric.accession] = metric.value
            elif isinstance(metric.value, list):
                for i, value in enumerate(metric.value):
                    metrics[set_label][f"{metric.accession}_{i}"] = value
            elif isinstance(metric.value, dict):
                for key, values in metric.value.items():
                    for i, value in enumerate(values):
                        metrics[set_label][
                            f"{metric.accession}_{key}_{i}"
                        ] = value

    return pd.DataFrame(metrics).T


# Read the outlier scores from all mzQC files.
filename_loop = "../data/Mtb-120-outlier-loop.mzQC"
with open(filename_loop) as f_in:
    mzqc_data = qc.JsonSerialisable.from_json(f_in)
outlier_scores_loop = pd.DataFrame(
    mzqc_data.setQualities[0].qualityMetrics[0].value
)
outlier_scores_loop = outlier_scores_loop.rename(columns={"MS:4000148": "QuaMeter LoOP"})

filename_pca = "../data/Mtb-120-outlier-pca.mzQC"
with open(filename_pca) as f_in:
    mzqc_data = qc.JsonSerialisable.from_json(f_in)
outlier_scores_pca = pd.DataFrame(
    mzqc_data.setQualities[0].qualityMetrics[0].value
)
# Scale PCA scores to [0, 1].
outlier_scores_pca["MS:4000148"] = (
    outlier_scores_pca["MS:4000148"] - outlier_scores_pca["MS:4000148"].min()) / (
        outlier_scores_pca["MS:4000148"].max() - outlier_scores_pca["MS:4000148"].min()
)
outlier_scores_pca = outlier_scores_pca.rename(columns={"MS:4000148": "QuaMeter PCA"})

# FIXME: This should take mzQC as input instead.
from outlier import *
metrics_ptxqc = pd.read_csv(
    "../data/report_v1.1.2_combined_heatmap.txt", sep="\t"
).set_index("fc.raw.file")
# Preprocess the data and compute outlier scores.
n_neighbors = 20
min_var = 0.01
min_corr = 0.9
scaling = "robust"
lmbd = 3
metric = "manhattan"
metrics_ptxqc = preprocess(metrics_ptxqc, min_var, min_corr, scaling)
outlier_scores_ptxqc = detect_outliers_loop(metrics_ptxqc, n_neighbors, lmbd, metric)
outlier_scores_ptxqc = pd.DataFrame(
    data={"MS:4000086": metrics_ptxqc.index, "MS:4000148": outlier_scores_ptxqc}
)
outlier_scores_ptxqc = outlier_scores_ptxqc.rename(columns={"MS:4000148": "PTXQC LoOP"})

outlier_scores = functools.reduce(
    lambda left, right: pd.merge(left, right, on="MS:4000086"),
    [outlier_scores_loop, outlier_scores_pca, outlier_scores_ptxqc],
)
outlier_scores = outlier_scores.set_index("MS:4000086")
outlier_scores["flagged"] = (outlier_scores.max(axis=1) > 0.66).astype(int)
outlier_scores["flagged"] += (outlier_scores.max(axis=1) > 0.33)

print(outlier_scores["flagged"].value_counts())
print(outlier_scores.sort_values("QuaMeter LoOP"))
print(outlier_scores.drop(columns=["flagged"]).corr("pearson"))

width = 4.5
height = width / 1.618
fig, ax = plt.subplots(figsize=(width, height))

pd.plotting.parallel_coordinates(
    outlier_scores.sort_values("flagged"),
    "flagged",
    ax=ax,
    color=["lightgray", "#dee000", "#ee266d"],
    axvlines=True,
    clip_on=False,
)

# ax.set_ylim(0, 1)
ax.set_ylabel("Outlier score")

ax.get_legend().remove()

plt.tight_layout()

plt.savefig("outlier_scores.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
