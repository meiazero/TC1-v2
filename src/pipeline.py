import os
from datetime import datetime

from config.parser import load_model_configs
from data.loader import load_raw_data, clean_data
from data.preprocess import preprocess
from training.trainer import train_and_evaluate
from training.evaluation import results_to_dataframe, select_best_model
from utils.io import make_dir, save_dataframe, save_model
from utils.logging import get_logger
import matplotlib.pyplot as plt
from plots.scatter import plot_actual_vs_predicted
from plots.residuals import plot_residuals
from plots.diagnostics import plot_qq

def _slugify_params(params: dict) -> str:
    """Create a filesystem-safe slug from parameter dict."""
    items = sorted(params.items())
    parts = []
    for k, v in items:
        s = str(v)
        for ch in [' ', '[', ']', ',', '(', ')', '\'', '"']:
            s = s.replace(ch, '')
        parts.append(f"{k}-{s}")
    return '__'.join(parts) if parts else 'default'

def run_pipeline(
    config_path: str,
    data_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    logger = get_logger(__name__)
    logger.info("Starting pipeline")

    # Setup output directory for this run
    make_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    make_dir(run_dir)
    # prepare directory for plots
    plots_dir = os.path.join(run_dir, "plots")
    make_dir(plots_dir)

    # Load and clean data
    logger.info("Loading raw data from %s", data_path)
    df_raw = load_raw_data(data_path)
    df = clean_data(df_raw)
    logger.info("Loaded data with %d records after cleaning", len(df))

    # Preprocess data
    logger.info("Preprocessing data")
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df, target_col="price", test_size=test_size, random_state=random_state
    )
    # Save scaler
    scaler_path = os.path.join(run_dir, "scaler.pkl")
    save_model(scaler, scaler_path)

    # Load model configurations
    logger.info("Loading model configurations from %s", config_path)
    experiments = load_model_configs(config_path)

    # Train, evaluate and plot results for each model
    results = []
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        params = exp["params"]
        model = exp["model"]
        logger.info("Training model %s with params %s", name, params)
        res = train_and_evaluate(model, name, params, X_train, y_train, X_test, y_test)
        results.append(res)
        # prepare parameter identifiers for filenames and titles
        param_slug = _slugify_params(params)
        param_str = ", ".join([f"{k}={v}" for k, v in sorted(params.items())]) if params else "default"

        # generate diagnostic plots for train and test sets, organized by model
        model_plots_dir = os.path.join(plots_dir, name)
        make_dir(model_plots_dir)
        for split_name in ("train", "test"):
            try:
                data = res.get(split_name, {})
                y_true = data.get("y_true")
                y_pred = data.get("y_pred")
                residuals = data.get("residuals")
                if y_true is None or y_pred is None or residuals is None:
                    continue
                # Actual vs Predicted scatter with R2, trend line, and param info
                fig, ax = plot_actual_vs_predicted(
                    y_true, y_pred,
                    model_name=name,
                    split_name=split_name
                )
                fig.suptitle(f"Params: {param_str}", fontsize=8)
                fig.savefig(os.path.join(
                    model_plots_dir,
                    f"{idx}_{param_slug}_{split_name}_actual_vs_predicted.png"
                ))
                plt.close(fig)

                # Residuals histogram with normal overlay and param info
                fig, ax = plot_residuals(
                    y_true, y_pred,
                    model_name=name,
                    split_name=split_name
                )
                fig.suptitle(f"Params: {param_str}", fontsize=8)
                fig.savefig(os.path.join(
                    model_plots_dir,
                    f"{idx}_{param_slug}_{split_name}_residuals_histogram.png"
                ))
                plt.close(fig)

                # QQ-Plot of residuals with param info
                fig, ax = plot_qq(
                    residuals,
                    model_name=name,
                    split_name=split_name
                )
                fig.suptitle(f"Params: {param_str}", fontsize=8)
                fig.savefig(os.path.join(
                    model_plots_dir,
                    f"{idx}_{param_slug}_{split_name}_qqplot.png"
                ))
                plt.close(fig)
            except Exception as e:
                logger.warning(
                    "Could not generate %s plots for %s: %s",
                    split_name, name, e
                )

    # Consolidate results
    df_results = results_to_dataframe(results)
    results_path = os.path.join(run_dir, "results.csv")
    save_dataframe(df_results, results_path)

    # Select and save best model
    best = select_best_model(df_results, metric="r2_test")
    if bool(best):
        logger.info("Best model: %s", best)
        # Find corresponding model instance
        for exp, res in zip(experiments, results):
            if res.get("model") == best["model"] and res.get("params") == best["params"]:
                best_model = exp["model"]
                break
        else:
            best_model = None
        if best_model:
            best_model_path = os.path.join(run_dir, "best_model.pkl")
            save_model(best_model, best_model_path)

    logger.info("Pipeline finished. Outputs are in %s", run_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ML pipeline for real estate dataset")
    parser.add_argument(
        "--config", default="src/config/models.yml", help="Path to model config file"
    )
    parser.add_argument(
        "--data", default="data/raw/real-estate-valuation-dataset.csv", help="Path to raw data CSV"
    )
    parser.add_argument(
        "--output", default="experiments", help="Directory to store experiment outputs"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )
    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )