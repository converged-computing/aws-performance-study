#!/usr/bin/env python3

import re
import numpy
import argparse
import sys
import json
import os
import pandas
import shap
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import matplotlib.pyplot as plt
import seaborn as sns

here = os.path.dirname(os.path.abspath(__file__))
analysis_root = os.path.dirname(here)
root = os.path.dirname(analysis_root)
sys.path.insert(0, analysis_root)

import performance_study as ps

# Each filename has one futex value and cpu metrics for all three iterations.
futex_lookup = {"waiting": {}}
cpu_lookup = {"waiting": {}, "running": {}}
thread_lookup = {}
proc_lookup = {}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--root",
        help="root directory with experiments",
        default=os.path.join(root, "experiment"),
    )
    parser.add_argument(
        "--out",
        help="directory to save parsed results",
        default=os.path.join(here, "data"),
    )
    return parser


def main():
    """
    Find application result files to parse.
    """
    global db
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Output images and data
    outdir = os.path.abspath(args.out)
    indir = os.path.abspath(args.root)

    # We absolutely want on premises results here
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Find input files (skip anything with test)
    parse_data(indir, outdir)


def parse_cpu(item):
    """
    Parse CPU
    """
    item = item.split("Cleaning up BPF resources")[-1]
    lines = [x for x in item.split("\n") if x and "Possibly lost" not in x][1:-1]
    item = "\n".join(lines)
    models = json.loads(item.split("Initiating cleanup sequence...")[-1])
    for model_type, model_names in models.items():
        for model in model_names:

            # E.g,. migration/22, kworker/88:2
            command = model["comm"].split("/")[0]
            if command != "xhpcg":
                continue

            # Add the median for now
            try:
                time_waiting_q = (
                    model["runq_latency_stats_ns"]["median_ns"]
                    * model["runq_latency_stats_ns"]["count"]
                )
            except:
                continue
            if time_waiting_q is not None:
                yield "cpu_waiting_ns", time_waiting_q, command
            try:
                time_running = (
                    model["on_cpu_stats_ns"]["median_ns"]
                    * model["on_cpu_stats_ns"]["count"]
                )
            except:
                time_running = None
            if time_running is not None:
                yield "cpu_running_ns", time_running, command


def parse_futex(item):
    """
    Futex parsing
    """
    item = item.split("Cleaning up BPF resources")[-1]
    lines = [x for x in item.split("\n") if x][1:-1]
    item = "\n".join(lines)
    models = json.loads(item)
    for model_type, model_names in models.items():
        for model in model_names:
            if model["comm"] != "xhpcg":
                continue
            median = model["wait_duration_stats_ns"]["median"]
            count = 0
            for futex_id, increment in model["futex_op_counts"].items():
                count += increment
            yield "median_futex_wait", median * count, model["comm"]


def load_node_features():
    """
    Read in NFD features.
    """
    feature_file = os.path.abspath(
        os.path.join(here, "../../docs/node-explorer/node-features.json")
    )
    valid_clusters = os.listdir("../../experiment/on-premises/results/logs")
    features = ps.read_json(feature_file)
    feature_dir = "../../experiment/on-premises/results/features"
    for feature_file in os.listdir(feature_dir):
        if not feature_file.startswith("labels"):
            continue
        on_prem_features = ps.read_json(os.path.join(feature_dir, feature_file))
        cluster = feature_file.replace("labels-", "").replace(".json", "")
        if cluster not in valid_clusters:
            continue
        on_prem_features["node.kubernetes.io/instance-type"] = cluster
        features.append(on_prem_features)

    unique_features = set()
    feature_values = {}

    for featset in features:
        for feature, feature_value in featset.items():
            if feature.startswith("feature.node"):
                unique_features.add(feature)
            else:
                continue
            if feature not in feature_values:
                feature_values[feature] = set()
            feature_values[feature].add(feature_value)

    feature_keepers = {}

    skip_features = "(%s)" % "|".join(["os_release.VERSION_ID.minor"])
    for feature, feature_options in feature_values.items():
        if len(feature_options) <= 1 or re.search(skip_features, feature):
            continue
        feature_keepers[feature] = sorted(list(feature_options))

    columns = []
    for feature, feature_values in feature_keepers.items():
        for feature_value in feature_values:
            columns.append(f"{feature}_{feature_value}")

    features = {
        x["node.kubernetes.io/instance-type"].split(".")[0]: x for x in features
    }
    return features, list(feature_keepers)


def add_plot_metrics(metrics, ax):
    x_text_coord = 0.95
    y_text_start = 0.95
    y_text_offset = 0.07
    i = 0
    for label, value in metrics.items():
        text_string = f"{label}: {value:.3f}"
        current_y_coord = y_text_start - (i * y_text_offset)
        ax.text(
            x_text_coord,
            current_y_coord,
            text_string,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8, alpha=0.7
            ),
        )
        i += 1


def parse_data(indir, outdir):
    """
    Parse filepaths for environment, etc., and results files for data.
    """
    features, columns = load_node_features()

    columns += ["optimization", "threads", "micro_arch"]
    columns += ["futex_waiting_ns", "cpu_running_ns", "cpu_waiting_ns"]

    x_files = ["hpcg_processes.csv", "hpcg_threads_per_process.csv"]
    y_files = [
        "hpcg_mpi_allreduce_avg.csv",
        "hpcg_fom.csv",
        "hpcg_total_cg_iterations.csv",
        "hpcg_memory_used_data_total_gbytes.csv",
        "hpcg_memory_bandwidth_across_kernels_write.csv",
        "hpcg_gflops_per_second_ddot.csv",
        "hpcg_gflops_per_second_mg.csv",
        "hpcg_setup_time_seconds.csv",
        "hpcg_memory_bandwidth_across_kernels_read.csv",
        "hpcg_memory_bandwidth_across_kernels_total.csv",
        "hpcg_gflops_per_second_spmv.csv",
        "hpcg_mpi_allreduce_max.csv",
        "hpcg_duration.csv",
        "hpcg_mpi_allreduce_min.csv",
        "hpcg_gflops_per_second_waxpby.csv",
        "hpcg_fom_per_dollar.csv",
    ]

    for filename in x_files:
        x_df = pandas.read_csv(os.path.join(here, "data", "heatmap", "csv", filename))
        for _, row in x_df.iterrows():
            if row.metric == "compatible":
                continue
            family = row.env.split(".")[0]
            if row.filename not in futex_lookup and "on-premises" not in row.filename:
                futex_lookup[row.filename] = {}
                cpu_lookup[row.filename] = {}
                futex_item = ps.read_file(row.filename.replace("hpcg.out", "futex.out"))
                cpu_item = ps.read_file(row.filename.replace("hpcg.out", "cpu.out"))
                for name, value, _ in parse_futex(futex_item):
                    futex_lookup[row.filename][name] = value
                for name, value, _ in parse_cpu(cpu_item):
                    cpu_lookup[row.filename][name] = value
            if row.metric == "processes":
                proc_lookup[family] = row.value
            elif row.metric == "threads_per_process":
                thread_lookup[family] = row.value

    numerical_columns = [
        "threads",
        "futex_waiting_ns",
        "cpu_waiting_ns",
        "cpu_running_ns",
    ]
    categorical_columns = [x for x in columns if x not in numerical_columns]

    models_dir = os.path.join(outdir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    top_features_json = {}
    interface_data = []
    seen = set()

    for filename in y_files:
        df = pandas.DataFrame(columns=columns)
        y_df = pandas.read_csv(os.path.join(here, "data", "heatmap", "csv", filename))
        y_actual = []
        for _, row in y_df.iterrows():
            if row.metric == "compatible":
                continue
            if "dollar" in filename:
                row.metric = "fom_per_dollar"
            family = row.env.split(".")[0]
            if family not in features:
                if family not in seen:
                    print(f"{family} missing from features - Vanessa add it!")
                    seen.add(family)
                continue
            y_actual.append(float(row.value))
            feature_vector = []
            for column in columns:
                if column.startswith("feature.node"):
                    feature_vector.append(features[family][column])
            opt, micro_arch = row.problem_size.rsplit("-", 1)
            cpu_running, cpu_waiting, futex_time = None, None, None
            if "on-premises" not in row.filename:
                cpu_running = cpu_lookup[row.filename].get("cpu_running_ns")
                cpu_waiting = cpu_lookup[row.filename].get("cpu_waiting_ns")
                futex_time = futex_lookup[row.filename].get("median_futex_wait")
            feature_vector += [
                opt,
                thread_lookup[family],
                micro_arch,
                futex_time,
                cpu_running,
                cpu_waiting,
            ]
            iteration = row.iteration
            uid = f"{row.env}.{row.problem_size}.{iteration}"
            while uid in df.index:
                iteration += 1
                uid = f"{row.env}.{row.problem_size}.{iteration}"
            df.loc[uid, columns] = feature_vector

        preprocessor_lr = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", drop="first"),
                    categorical_columns,
                ),
                ("num", StandardScaler(), numerical_columns),
            ]
        )

        preprocessor_rf = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", drop="first"),
                    categorical_columns,
                ),
                ("num", "passthrough", numerical_columns),
            ]
        )

        df["y"] = y_actual
        df = df.dropna()
        y_actual = df["y"]
        df = df.drop("y", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            df, y_actual, test_size=0.2, random_state=42
        )

        title = " ".join([x.capitalize() for x in row.metric.split("_")])

        # --- Linear Regression ---
        pipeline_lr = Pipeline(
            steps=[("preprocessor", preprocessor_lr), ("regressor", LinearRegression())]
        )
        pipeline_lr.fit(X_train, y_train)
        y_pred_lr = pipeline_lr.predict(X_test)
        metrics_lr = {
            "MAE": mean_absolute_error(y_test, y_pred_lr),
            "MSE": mean_squared_error(y_test, y_pred_lr),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred_lr),
            "R2 Score": r2_score(y_test, y_pred_lr),
        }
        print(f"\n--- {row.metric} ---")
        print(
            f"Linear Regression - MAE: {metrics_lr['MAE']:.2f}, MSE: {metrics_lr['MSE']:.2f}, R2: {metrics_lr['R2 Score']:.2f}, MAPE: {metrics_lr['MAPE']:.2f}"
        )

        plt.figure(figsize=(8, 6))
        ax = sns.regplot(
            x=y_test, y=y_pred_lr, scatter_kws={"alpha": 0.3}, line_kws={"color": "red"}
        )
        plt.title(f"Linear Regression: Actual vs. Predicted ({title})")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        add_plot_metrics(metrics_lr, ax)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(models_dir, f"xhpcg_{row.metric}_linear_regression.svg")
        )
        plt.clf()
        plt.close()

        # --- Random Forest Regressor ---
        pipeline_rf = Pipeline(
            steps=[
                ("preprocessor", preprocessor_rf),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )
        pipeline_rf.fit(X_train, y_train)
        y_pred_rf = pipeline_rf.predict(X_test)
        metrics_rf = {
            "MAE": mean_absolute_error(y_test, y_pred_rf),
            "MSE": mean_squared_error(y_test, y_pred_rf),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred_lr),
            "R2 Score": r2_score(y_test, y_pred_rf),
        }
        print(
            f"Random Forest - MAE: {metrics_rf['MAE']:.2f}, MSE: {metrics_rf['MSE']:.2f}, R2: {metrics_rf['R2 Score']:.2f}, MAPE: {metrics_rf['MAPE']:.2f}"
        )

        plt.figure(figsize=(8, 6))
        ax = sns.regplot(
            x=y_test,
            y=y_pred_rf,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "green"},
        )
        plt.title(f"Random Forest: Actual vs. Predicted ({title})")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        add_plot_metrics(metrics_rf, ax)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, f"xhpcg_{row.metric}_random_forest.svg"))
        plt.clf()
        plt.close()

        # --- LASSO Regression
        # Use the same preprocessor as Linear Regression (with StandardScaler)
        pipeline_lasso = Pipeline(
            steps=[
                ("preprocessor", preprocessor_lr),
                # Alpha is the regularization strength. Higher values lead to more feature coefficients being zero.
                ("regressor", Lasso(alpha=0.1, random_state=42)),
            ]
        )
        pipeline_lasso.fit(X_train, y_train)
        y_pred_lasso = pipeline_lasso.predict(X_test)
        metrics_lasso = {
            "MAE": mean_absolute_error(y_test, y_pred_lasso),
            "MSE": mean_squared_error(y_test, y_pred_lasso),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred_lr),
            "R2 Score": r2_score(y_test, y_pred_lasso),
        }
        print(
            f"LASSO Regression - MAE: {metrics_lasso['MAE']:.2f}, MSE: {metrics_lasso['MSE']:.2f}, R2: {metrics_lasso['R2 Score']:.2f}, MAPE: {metrics_lasso['MAPE']:.2f}"
        )

        plt.figure(figsize=(8, 6))
        ax = sns.regplot(
            x=y_test,
            y=y_pred_lasso,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "purple"},
        )
        plt.title(f"LASSO Regression: Actual vs. Predicted ({title})")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        add_plot_metrics(metrics_lasso, ax)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(models_dir, f"xhpcg_{row.metric}_lasso_regression.svg")
        )
        plt.clf()
        plt.close()

        df["y_actual"] = y_actual
        df.to_csv(os.path.join(models_dir, f"features_{filename}"))

        # SHAP Analysis for LASSO
        print(f"Generating SHAP values for LASSO model...")
        preprocessor = pipeline_lasso.named_steps["preprocessor"]
        regressor = pipeline_lasso.named_steps["regressor"]
        X_train_transformed_background = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        if hasattr(X_train_transformed_background, "toarray"):
            X_train_transformed_background = X_train_transformed_background.toarray()
        X_train_transformed_background = X_train_transformed_background.astype(
            numpy.float32
        )

        feature_names_out = preprocessor.get_feature_names_out()

        # Use LinearExplainer for LASSO, as it's a linear model
        explainer = shap.LinearExplainer(regressor, X_train_transformed_background)

        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()
        X_test_transformed = X_test_transformed.astype(numpy.float32)
        X_test_transformed_df = pandas.DataFrame(
            X_test_transformed, columns=feature_names_out
        )

        shap_values = explainer.shap_values(X_test_transformed_df)

        shap_sum = numpy.abs(shap_values).mean(axis=0)
        indices = numpy.argsort(shap_sum)[::-1]
        top_n_features = [feature_names_out[i] for i in indices[:3]]

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, X_test_transformed_df, show=False, plot_type="bar"
        )
        metric = " ".join(
            [x.capitalize() for x in row.metric.replace("_", " ").split(" ")]
        ).replace("fom", "FOM")
        plt.subplots_adjust(left=0.35, bottom=0.15, right=1.1)
        plt.title(
            f"SHAP Global Feature Importance for {metric} (LASSO)", loc="left"
        )  # <--- MODIFIED
        plt.savefig(
            os.path.join(models_dir, f"shap_summary_{row.metric}_bar.svg"),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_transformed_df, show=False)
        plt.title(
            f"SHAP Global Feature Importance for {metric} (LASSO)", loc="left"
        )  # <--- MODIFIED
        plt.subplots_adjust(left=0.35, bottom=0.15)
        plt.savefig(
            os.path.join(models_dir, f"shap_summary_{row.metric}_dot.svg"),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()
        interface_data.append(
            {
                "id": row.metric,
                "displayName": metric,
                "dotImagePath": f"shap_summary_{row.metric}_dot.svg",
                "barImagePath": f"shap_summary_{row.metric}_bar.svg",
            }
        )
        top_features = "\n" + "\n".join(top_n_features)
        print(f"Top SHAP features for {metric}: {top_features}")
        top_features_json[row.metric] = top_n_features

    ps.write_json(top_features_json, os.path.join(models_dir, "top-features.json"))
    ps.write_json(interface_data, os.path.join(models_dir, "metric-data.json"))


if __name__ == "__main__":
    main()
