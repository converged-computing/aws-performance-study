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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    # {'tgid': 3592,
    # 'comm': 'containerd',
    # 'cgroup_id': 6520,
    # 'wait_duration_stats_ns': {'count': 1693.0,
    #  'sum': 224436362166.99985,
    #  'mean': 132567254.67631415,
    #  'median': 221633.7996235165,
    #  'min': 812,
    #  'max': 25489918732,
    #  'variance_ns2': 9.336119483708768e+17,
    #  'iqr_ns': 97748409.18476921,
    #  'q1_ns': 65205.02079568948,
    #  'q3_ns': 97813614.2055649},
    # 'futex_op_counts': {'128': 1693},
    # 'first_seen_ts_ns': 1941370732957,
    # 'last_seen_ts_ns': 1975537850634}
    for model_type, model_names in models.items():
        for model in model_names:
            # Just concerned with app for now
            if model["comm"] != "xhpcg":
                continue
            # Add the median for now...
            median = model["wait_duration_stats_ns"]["median"]
            # The number of times
            count = 0
            for futex_id, increment in model["futex_op_counts"].items():
                count += increment
            yield "median_futex_wait", median * count, model["comm"]


def load_node_features():
    """
    Read in NFD features.
    """
    # First read in node features, get unique values
    # For each unique value, get all possible options
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

    # We also need to add the on-premises
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

    # Now we want to get rid of features that add no value
    feature_keepers = {}

    skip_features = "(%s)" % "|".join(["os_release.VERSION_ID.minor"])
    for feature, feature_options in feature_values.items():
        if len(feature_options) <= 1 or re.search(skip_features, feature):
            continue
        feature_keepers[feature] = sorted(list(feature_options))

    # Create the start of columns - assume one hot encoding
    # Note - found pandas can do this for me!
    columns = []
    for feature, feature_values in feature_keepers.items():
        for feature_value in feature_values:
            columns.append(f"{feature}_{feature_value}")

    # Make feature family into a lookup
    # TODO keep trying to run on dane / ruby / and get labels for corona190
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
        text_string = (
            f"{label}: {value:.3f}"  # Format the string (e.g., 3 decimal places)
        )

        # Calculate the current y-coordinate for this metric
        current_y_coord = y_text_start - (i * y_text_offset)
        ax.text(
            x_text_coord,
            current_y_coord,
            text_string,
            transform=ax.transAxes,  # Important: use axes coordinates (0 to 1)
            fontsize=12,
            verticalalignment="top",  # Align text from its top
            horizontalalignment="right",  # Align text from its right
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

    # We will also include optimization, cores, and micro-arch
    columns += ["optimization", "threads", "micro_arch"]

    # And futex wait, cpu running and waiting
    # TODO do we want to add ratio or too correlated?
    columns += ["futex_waiting_ns", "cpu_running_ns", "cpu_waiting_ns"]

    # For each csv of something we want to predict (Y) assemble data frame
    # These have been normalized to account for weak scaling
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

    # First generate lookups for futex, cpu, and threads/cores
    for filename in x_files:
        x_df = pandas.read_csv(os.path.join(here, "data", "heatmap", "csv", filename))
        for _, row in x_df.iterrows():
            if row.metric == "compatible":
                continue
            family = row.env.split(".")[0]

            # Use a lookup as we go. We don't have on-premises ebpf
            if row.filename not in futex_lookup and "on-premises" not in row.filename:

                # Assume we haven't seen futex or cpu
                futex_lookup[row.filename] = {}
                cpu_lookup[row.filename] = {}
                futex_item = ps.read_file(row.filename.replace("hpcg.out", "futex.out"))
                cpu_item = ps.read_file(row.filename.replace("hpcg.out", "cpu.out"))
                for name, value, _ in parse_futex(futex_item):
                    futex_lookup[row.filename][name] = value
                for name, value, _ in parse_cpu(cpu_item):
                    cpu_lookup[row.filename][name] = value

            # These are consistent across instance types
            # IMPORTANT - in this study we just do one size / type
            if row.metric == "processes":
                proc_lookup[family] = row.value
            elif row.metric == "threads_per_process":
                thread_lookup[family] = row.value

    # These features need one hot encoding
    # Don't include cores - we normalized data by count
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

    # Now generate models!
    for filename in y_files:
        # Here is our feature df
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

            # This is the Y to predict
            y_actual.append(float(row.value))

            # Once we get here, we have the values
            # Assemble the features for the node, we only need to know
            feature_vector = []
            for column in columns:
                if column.startswith("feature.node"):
                    # Look up feature by instance name
                    feature_vector.append(features[family][column])

            # Last set are:
            # opt, cores, threads, micro_arch, futex_waiting_ns, cpu_running_ns, cpu_waiting_ns, y_pred
            opt, micro_arch = row.problem_size.rsplit("-", 1)
            cpu_running = None
            cpu_waiting = None
            futex_time = None
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
            # on-premises runs have repeat iteration numbers for different ones
            while uid in df.index:
                iteration += 1
                uid = f"{row.env}.{row.problem_size}.{iteration}"
            df.loc[uid, columns] = feature_vector

        # When we get here, we need to hot one encode
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", drop="first"),
                    categorical_columns,
                ),
                (
                    "num",
                    StandardScaler(),
                    numerical_columns,
                ),  # passthrough would skip normalization
            ]
        )

        # Create a pipeline with preprocessing and a model - linear regression. I'm simple
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

        # TODO: consider not including eBPF if not meaningful because we have to drop
        df["y"] = y_actual
        df = df.dropna()
        y_actual = df["y"]
        df = df.drop("y", axis=1)

        # Split the data (using the original X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            df, y_actual, test_size=0.2, random_state=42
        )

        # Fit linear regression
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae_lr = mean_absolute_error(y_test, y_pred)
        mse_lr = mean_squared_error(y_test, y_pred)
        r2_lr = r2_score(y_test, y_pred)
        print(row.metric)
        print(f"Linear Regression (Unscaled) - MAE: {mae_lr:.2f}")
        print(f"Linear Regression (Unscaled) - MSE: {mse_lr:.2f}")
        print(f"Linear Regression (Unscaled) - R2 Score: {r2_lr:.2f}")

        # Plot predictions vs actuals
        metrics = {"MAE": mae_lr, "MSE": mse_lr, "R2 Score": r2_lr}
        title = " ".join([x.capitalize() for x in row.metric.split("_")])
        plt.figure(figsize=(8, 6))
        ax = sns.regplot(
            x=y_test, y=y_pred, scatter_kws={"alpha": 0.3}, line_kws={"color": "red"}
        )
        plt.title(f"Linear Regression: Actual vs. Predicted ({title})")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        add_plot_metrics(metrics, ax)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(models_dir, f"xhpcg_{row.metric}_linear_regression.svg")
        )
        plt.clf()
        plt.close()

        # Save linear model pipeline for shapley
        lm_pipeline = copy.deepcopy(pipeline)

        # Random forest
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(handle_unknown="ignore", drop="first"),
                                categorical_columns,
                            ),
                            ("num", "passthrough", numerical_columns),
                        ]
                    ),
                ),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mae_rf = mean_absolute_error(y_test, y_pred)
        mse_rf = mean_squared_error(y_test, y_pred)
        r2_rf = r2_score(y_test, y_pred)
        metrics = {"MAE": mae_rf, "MSE": mse_rf, "R2 Score": r2_rf}

        print(f"Random Forest Regressor - MAE: {mae_rf:.2f}")
        print(f"Random Forest Regressor - MSE: {mse_rf:.2f}")
        print(f"Random Forest Regressor - R2 Score: {r2_rf:.2f}")

        plt.figure(figsize=(8, 6))
        ax = sns.regplot(
            x=y_test, y=y_pred, scatter_kws={"alpha": 0.3}, line_kws={"color": "red"}
        )
        plt.title(f"Random Forest Regressor: Actual vs. Predicted ({title})")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        add_plot_metrics(metrics, ax)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, f"xhpcg_{row.metric}_random_forest.svg"))
        plt.clf()
        plt.close()

        # Save data with y_preds added
        df["y_actual"] = y_actual
        df.to_csv(os.path.join(models_dir, f"features_{filename}"))

        # Use the linear model - trees are more likely to overfit I think
        preprocessor = lm_pipeline.named_steps["preprocessor"]
        regressor = lm_pipeline.named_steps["regressor"]
        X_train_transformed_background = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # This will issue an error later if it isn't a float array
        X_train_transformed_background = X_train_transformed_background.toarray()
        X_train_transformed_background = X_train_transformed_background.astype(
            numpy.float32
        )

        feature_names_out = preprocessor.get_feature_names_out()

        # Which features are meaningful?
        explainer = shap.LinearExplainer(regressor, X_train_transformed_background)
        X_test_transformed = X_test_transformed.toarray()
        X_test_transformed = X_test_transformed.astype(numpy.float32)
        X_test_transformed_df = pandas.DataFrame(
            X_test_transformed, columns=feature_names_out
        )

        # DRUMROLL!
        shap_values = explainer.shap_values(X_test_transformed_df)

        # Get relative contribution of top features
        shap_sum = numpy.abs(shap_values).mean(axis=0)
        indices = numpy.argsort(shap_sum)[::-1]
        top_n_features = [feature_names_out[i] for i in indices[:3]]

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, X_test_transformed_df, show=False, plot_type="bar"
        )
        metric = " ".join(
            [x.capitalize() for x in row.metric.replace("_", " ").split(" ")]
        )
        metric = metric.replace("fom", "FOM")
        plt.subplots_adjust(left=0.35, bottom=0.15, right=1.1)
        plt.title(f"SHAP Global Feature Importance for {metric}", loc="left")
        plt.savefig(
            os.path.join(models_dir, f"shap_summary_{row.metric}_bar.svg"),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_transformed_df, show=False)
        plt.title(f"SHAP Global Feature Importance for {metric}", loc="left")
        plt.subplots_adjust(left=0.35, bottom=0.15)
        plt.savefig(
            os.path.join(models_dir, f"shap_summary_{row.metric}_dot.svg"),
            bbox_inches="tight",
        )
        plt.clf()
        plt.close()

        #shap_interaction_values = tree_explainer.shap_interaction_values(X_test)
        #plt.figure(figsize=(12, 10))
        ##shap.summary_plot(shap_interaction_values, X_test_transformed_df)
        #plt.subplots_adjust(left=0.35, bottom=0.15)
        #plt.savefig(
        #    os.path.join(models_dir, f"shap_summary_{row.metric}_interactions.svg"),
        #    bbox_inches="tight",
        #)
        #plt.clf()
        #plt.close()
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
        # for feature_name in top_n_features:
        #    plt.figure()
        #    shap.dependence_plot(feature_name, shap_values, X_test_transformed_df, interaction_index="auto", show=False)
        #    # You can also specify a particular feature name for `interaction_index`.
        #    plt.title(f"SHAP Dependence Plot for {metric} and '{feature_name}'")
        #    plt.tight_layout()
        #    plt.savefig(os.path.join(models_dir, f"shap_dependence_{feature_name.replace(' ', '_').replace('/', '_')}.png"))
        #    plt.clf()
        #    plt.close()

    ps.write_json(top_features_json, os.path.join(models_dir, "top-features.json"))
    ps.write_json(interface_data, os.path.join(models_dir, "metric-data.json"))


if __name__ == "__main__":
    main()
