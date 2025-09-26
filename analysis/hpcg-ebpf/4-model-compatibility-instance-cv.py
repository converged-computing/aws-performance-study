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

    columns += ["optimization", "threads", "memory_gib", "cores", "cost", "micro_arch"]
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
        "memory_gib",
        "cores",
        "cost",
        "futex_waiting_ns",
        "cpu_waiting_ns",
        "cpu_running_ns",
    ]
    categorical_columns = [x for x in columns if x not in numerical_columns]

    models_dir = os.path.join(outdir, "models-cv")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    top_features_json = {}
    interface_data = []
    seen = set()
    prediction_results = []

    # Here we need to leave out each instance type, so keep track of their indices
    for filename in y_files:
        df = pandas.DataFrame(columns=columns)
        if filename != "hpcg_fom.csv":
            continue
        y_df = pandas.read_csv(os.path.join(here, "data", "heatmap", "csv", filename))
        y_actual = []
        instance_types = []
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
            instance_types.append(features[family]["node.kubernetes.io/instance-type"])
            feature_vector += [
                opt,
                thread_lookup[family],
                ps.memory_lookup[row.env],
                ps.core_lookup[row.env],
                ps.cost_lookup[row.env],
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

        # Handle y and instance types
        df["y"] = y_actual
        df["instance_types"] = instance_types
        df = df.dropna()
        y_actual = df["y"]
        instance_types = df["instance_types"]
        df = df.drop("y", axis=1)
        df = df.drop("instance_types", axis=1)

        # Do analysis leaving out and using as test each instance type
        fom_df = pandas.DataFrame(
            columns=[
                "instance_type",
                "model",
                "test_train_lo_model_r2",
                "instance_predict_r2",
            ]
        )
        fom_idx = 0
        for instance_type in set(instance_types.tolist()):
            subset_test_idx = [
                i for i, x in enumerate(instance_types) if x == instance_type
            ]
            # This shouldn't happen
            if not subset_test_idx:
                continue
            subset_idx = [i for i, x in enumerate(instance_types) if x != instance_type]
            y_actual_subset = y_actual.iloc[subset_idx]
            y_actual_subset_test = y_actual.iloc[subset_test_idx]
            subset = df.iloc[subset_idx]
            subset_test = df.iloc[subset_test_idx]
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
            X_train, X_test, y_train, y_test = train_test_split(
                subset, y_actual_subset, test_size=0.2, random_state=42
            )
            title = " ".join([x.capitalize() for x in row.metric.split("_")])

            # --- Linear Regression ---
            pipeline_lr = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_lr),
                    ("regressor", LinearRegression()),
                ]
            )
            pipeline_lr.fit(X_train, y_train)
            y_pred_lr = pipeline_lr.predict(X_test)
            metrics_lr = {
                "MAE": mean_absolute_error(y_test, y_pred_lr),
                "MSE": mean_squared_error(y_test, y_pred_lr),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred_lr),
                "R2 Score": r2_score(y_test, y_pred_lr),
            }
            r2_model = metrics_lr["R2 Score"]
            print(f"\n--- {row.metric} leaving out {instance_type}---")
            print(
                f"Linear Regression - MAE: {metrics_lr['MAE']:.2f}, MSE: {metrics_lr['MSE']:.2f}, R2: {metrics_lr['R2 Score']:.2f}, MAPE: {metrics_lr['MAPE']:.2f}"
            )

            y_pred_lr = pipeline_lr.predict(subset_test)
            metrics_lr = {
                "MAE": mean_absolute_error(y_actual_subset_test, y_pred_lr),
                "MSE": mean_squared_error(y_actual_subset_test, y_pred_lr),
                "MAPE": mean_absolute_percentage_error(y_actual_subset_test, y_pred_lr),
                "R2 Score": r2_score(y_actual_subset_test, y_pred_lr),
            }

            # Save results
            for actual, pred in zip(y_actual_subset_test, y_pred_lr):
                prediction_results.append(
                    {
                        "instance_type": instance_type,
                        "model": "Linear Regression",
                        "y_actual": actual,
                        "y_predicted": pred,
                    }
                )
            print(
                f"\n--- {row.metric} predicting {instance_type} with left out model---"
            )
            fom_df.loc[fom_idx, :] = [
                instance_type,
                "linear-regression",
                r2_model,
                metrics_lr["R2 Score"],
            ]
            fom_idx += 1
            print(
                f"Linear Regression - MAE: {metrics_lr['MAE']:.2f}, MSE: {metrics_lr['MSE']:.2f}, R2: {metrics_lr['R2 Score']:.2f}, MAPE: {metrics_lr['MAPE']:.2f}"
            )

            # --- Random Forest Regressor ---
            pipeline_rf = Pipeline(
                steps=[
                    ("preprocessor", preprocessor_rf),
                    (
                        "regressor",
                        RandomForestRegressor(n_estimators=100, random_state=42),
                    ),
                ]
            )
            pipeline_rf.fit(X_train, y_train)
            y_pred_rf = pipeline_rf.predict(X_test)
            metrics_rf = {
                "MAE": mean_absolute_error(y_test, y_pred_rf),
                "MSE": mean_squared_error(y_test, y_pred_rf),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred_rf),
                "R2 Score": r2_score(y_test, y_pred_rf),
            }
            r2_model = metrics_rf["R2 Score"]
            print(f"\n--- {row.metric} leaving out {instance_type}---")
            print(
                f"Random Forest - MAE: {metrics_rf['MAE']:.2f}, MSE: {metrics_rf['MSE']:.2f}, R2: {metrics_rf['R2 Score']:.2f}, MAPE: {metrics_rf['MAPE']:.2f}"
            )
            y_pred_rf = pipeline_rf.predict(subset_test)
            metrics_rf = {
                "MAE": mean_absolute_error(y_actual_subset_test, y_pred_rf),
                "MSE": mean_squared_error(y_actual_subset_test, y_pred_rf),
                "MAPE": mean_absolute_percentage_error(y_actual_subset_test, y_pred_rf),
                "R2 Score": r2_score(y_actual_subset_test, y_pred_rf),
            }
            print(
                f"\n--- {row.metric} predicting {instance_type} with left out model---"
            )
            fom_df.loc[fom_idx, :] = [
                instance_type,
                "random-forest",
                r2_model,
                metrics_rf["R2 Score"],
            ]
            fom_idx += 1
            print(
                f"Random Forest - MAE: {metrics_rf['MAE']:.2f}, MSE: {metrics_rf['MSE']:.2f}, R2: {metrics_rf['R2 Score']:.2f}, MAPE: {metrics_rf['MAPE']:.2f}"
            )
            for actual, pred in zip(y_actual_subset_test, y_pred_rf):
                prediction_results.append(
                    {
                        "instance_type": instance_type,
                        "model": "Random Forest",
                        "y_actual": actual,
                        "y_predicted": pred,
                    }
                )

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
                "MAPE": mean_absolute_percentage_error(y_test, y_pred_lasso),
                "R2 Score": r2_score(y_test, y_pred_lasso),
            }
            r2_model = metrics_lasso["R2 Score"]
            print(f"\n--- {row.metric} leaving out {instance_type}---")
            print(
                f"LASSO Regression - MAE: {metrics_lasso['MAE']:.2f}, MSE: {metrics_lasso['MSE']:.2f}, R2: {metrics_lasso['R2 Score']:.2f}, MAPE: {metrics_lasso['MAPE']:.2f}"
            )

            y_pred_lasso = pipeline_lasso.predict(subset_test)
            metrics_lasso = {
                "MAE": mean_absolute_error(y_actual_subset_test, y_pred_lasso),
                "MSE": mean_squared_error(y_actual_subset_test, y_pred_lasso),
                "MAPE": mean_absolute_percentage_error(
                    y_actual_subset_test, y_pred_lasso
                ),
                "R2 Score": r2_score(y_actual_subset_test, y_pred_lasso),
            }
            print(
                f"\n--- {row.metric} predicting {instance_type} with left out model---"
            )
            fom_df.loc[fom_idx, :] = [
                instance_type,
                "lasso",
                r2_model,
                metrics_lasso["R2 Score"],
            ]
            fom_idx += 1
            print(
                f"LASSO Regression - MAE: {metrics_lasso['MAE']:.2f}, MSE: {metrics_lasso['MSE']:.2f}, R2: {metrics_lasso['R2 Score']:.2f}, MAPE: {metrics_lasso['MAPE']:.2f}"
            )
            for actual, pred in zip(y_actual_subset_test, y_pred_lasso):
                prediction_results.append(
                    {
                        "instance_type": instance_type,
                        "model": "LASSO Regression",
                        "y_actual": actual,
                        "y_predicted": pred,
                    }
                )
        fom_df = fom_df.sort_values(by="instance_predict_r2", ascending=False)
        fom_df.to_csv(
            os.path.join(models_dir, f"model_r2_complete_withcost_{filename}")
        )
        print(fom_df)

        # Make a plot to look at prediction errors
        pred_df = pandas.DataFrame(prediction_results)

        # 2. Calculate the prediction error (difference)
        pred_df["prediction_error"] = pred_df["y_predicted"] - pred_df["y_actual"]
        order_x = (
            pred_df.groupby("instance_type")["prediction_error"]
            .median()
            .sort_values()
            .index
        )
        hue_order = (
            pred_df.groupby("instance_type")["prediction_error"]
            .median()
            .sort_values()
            .index
        )

        plt.figure(figsize=(16, 9))

        # Use a boxplot to show the distribution of errors for each model
        # across all the left-out instance types.
        sns.boxplot(
            data=pred_df,
            x="model",
            y="prediction_error",
            hue="instance_type",
            hue_order=hue_order,
        )

        # Add a horizontal line at y=0 to represent a perfect prediction
        plt.axhline(
            0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Perfect Prediction (Error = 0)",
        )
        plt.title("Prediction Error Distribution on Left-Out Instances", fontsize=18)
        plt.xlabel("", fontsize=14)
        plt.ylabel("Prediction Error (Predicted FOM - Actual FOM)", fontsize=14)
        plt.xticks(fontsize=12)
        plt.legend()
        plt.tight_layout()

        # Save and show the plot
        plot_path = os.path.join(
            models_dir,
            f"prediction_error_boxplot_complete_withcost_{filename.replace('.csv', '.png')}",
        )
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # We'll create a single figure with three subplots, one for each model.
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
        fig.suptitle(
            "Predicted vs. Actual FOM on Left-Out Instances", fontsize=20
        )  # y parameter is often not needed with this fix

        models = ["Linear Regression", "Random Forest", "LASSO Regression"]
        for i, model_name in enumerate(models):
            ax = axes[i]

            # Filter the DataFrame for the current model
            model_df = pred_df[pred_df["model"] == model_name]

            # Create the scatter plot, coloring by instance type
            sns.scatterplot(
                data=model_df,
                x="y_actual",
                y="y_predicted",
                hue="instance_type",
                alpha=0.7,
                s=50,  # Marker size
                ax=ax,
            )

            # Add the y=x line for reference (perfect prediction)
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
            ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Perfect Prediction")

            ax.set_title(model_name, fontsize=16)
            ax.set_xlabel("Actual FOM", fontsize=12)
            ax.set_ylabel("Predicted FOM", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(title="Instance Type", fontsize="small")

        # The value [0, 0, 1, 0.96] leaves a 4% margin at the top for the suptitle.
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(
            models_dir,
            f"predicted_vs_actual_scatter_complete_withcost_{filename.replace('.csv', '.png')}",
        )
        plt.savefig(plot_path, dpi=300)
        print(f"Predicted vs. Actual scatter plot saved to {plot_path}")
        import IPython

        IPython.embed()
        plt.show()

        sys.exit()
        # Make a plot to look at prediction errors
        pred_df = pandas.DataFrame(prediction_results)

        # 2. Calculate the prediction error (difference)
        pred_df["prediction_error"] = pred_df["y_predicted"] - pred_df["y_actual"]
        order_x = (
            pred_df.groupby("instance_type")["prediction_error"]
            .median()
            .sort_values()
            .index
        )
        hue_order = (
            pred_df.groupby("instance_type")["prediction_error"]
            .median()
            .sort_values()
            .index
        )

        fig = plt.figure(figsize=(12, 3.3))
        gs = plt.GridSpec(1, 2, width_ratios=[2, 1])
        axes = []
        cpu_ax = fig.add_subplot(gs[0, 0])
        axes.append(cpu_ax)
        axes.append(fig.add_subplot(gs[0, 1], sharey=cpu_ax))

        # Use a boxplot to show the distribution of errors for each model
        # across all the left-out instance types.
        sns.boxplot(
            ax=axes[0],
            data=pred_df,
            x="model",
            y="prediction_error",
            hue="instance_type",
        )

        # Add a horizontal line at y=0 to represent a perfect prediction
        axes[0].axhline(
            0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Perfect Prediction (Error = 0)",
        )
        axes[0].set_title(
            "Prediction Error Distribution on Left-Out Instances", fontsize=18
        )
        axes[0].set_xlabel("Regression Model", fontsize=14)
        axes[0].set_ylabel("Prediction Error (Predicted FOM - Actual FOM)", fontsize=14)
        axes[0].legend()
        handles, labels = axes[0].get_legend_handles_labels()
        legend = axes[1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(-0.1, 0.5),
            frameon=False,
        )
        legend._ncol = 2  # Set the number of columns to 2
        for ax in axes[0:1]:
            ax.get_legend().remove()
        axes[1].axis("off")

        plt.tight_layout()

        # Save and show the plot
        plot_path = os.path.join(
            models_dir,
            f"prediction_error_boxplot_complete_withcost_{filename.replace('.csv', '.png')}",
        )
        plt.savefig(plot_path, dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
