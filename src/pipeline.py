import os
from datetime import datetime

from config.parser import load_model_configs
from data.loader import load_raw_data, clean_data
from data.preprocess import preprocess, split_data, remove_outliers as remove_outliers_fn
from training.trainer import train_and_evaluate
from training.evaluation import results_to_dataframe, select_best_model
from utils.io import make_dir, save_dataframe, save_model
from utils.logging import get_logger
from plots.boxplot import plot_summary_boxplots
import matplotlib.pyplot as plt
from plots.scatter import plot_actual_vs_predicted
from utils.statistics import significance_test
from plots.residuals import plot_residuals
from plots.learning_curve import plot_learning_curve

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
    random_state: int = 42,
    remove_outliers: bool = False,
    epochs: int = 1
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
    if remove_outliers:
        logger.info("Removing outliers from data")
        df = remove_outliers_fn(df)
        logger.info("Data shape after outlier removal: %d records", len(df))

    # Preprocess data
    logger.info("Preprocessing data")
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df, target_col="price", test_size=test_size, random_state=random_state
    )
    # Retain original train/test splits for significance testing
    X_train_df, X_test_df, y_train_series, y_test_series = split_data(
        df, target_col="price", test_size=test_size, random_state=random_state
    )
    # Save scaler
    scaler_path = os.path.join(run_dir, "scaler.pkl")
    save_model(scaler, scaler_path)

    # Load model configurations
    logger.info("Loading model configurations from %s", config_path)
    experiments = load_model_configs(config_path)

    # Train, evaluate (with repeats) and plot only the best run per model
    results = []
    for idx, exp in enumerate(experiments):
        name = exp["name"]
        params = exp["params"]
        model = exp["model"]
        logger.info("Training model %s with params %s", name, params)
        best_res = None

        # Repeat training to select best run
        for epoch in range(epochs):
            logger.info("Run %d/%d for %s", epoch + 1, epochs, name)
            res = train_and_evaluate(model, name, params, X_train, y_train, X_test, y_test)
            if "error" in res:
                logger.warning("Error during %s run %d: %s", name, epoch + 1, res.get("error"))
                continue
            if best_res is None or res.get("test", {}).get("r2", float("-inf")) > best_res.get("test", {}).get("r2", float("-inf")):
                best_res = res
        if best_res is None:
            logger.warning("All runs failed for %s, skipping", name)
            continue

        # Append only the best result for this model config
        results.append(best_res)
        res = best_res

    # Consolidate results and sort by test R2 descending
    df_results = results_to_dataframe(results)
    df_results = df_results.sort_values(by='r2_test', ascending=False).reset_index(drop=True)

    # Save full results table
    results_path = os.path.join(run_dir, "results.csv")
    save_dataframe(df_results, results_path)
    logger.info("Saved full results to %s", results_path)
    # Save model ranking by best experiments and generate ranking plot
    try:
        # Select best experiment per model based on highest test R2
        best_configs = df_results.loc[df_results.groupby('model')['r2_test'].idxmax()].copy()
        # Extract key metrics and residual variance (errors variability)
        model_rank = best_configs[[
            'model', 'r2_train', 'r2_test', 'rmse_train', 'rmse_test',
            'res_var_train', 'res_var_test'
        ]]
        # Rename residual variance columns
        model_rank = model_rank.rename(columns={
            'res_var_train': 'variance_train',
            'res_var_test': 'variance_test'
        })
        # Sort by test R2 descending
        model_rank = model_rank.sort_values(by='r2_test', ascending=False).reset_index(drop=True)
        # Save CSV
        rank_csv = os.path.join(run_dir, 'model_ranking.csv')
        save_dataframe(model_rank, rank_csv)
        logger.info("Saved model ranking to %s", rank_csv)

        # Bar plot for model ranking by Test R2
        # fig, ax = plt.subplots(figsize=(max(6, len(model_rank) * 1.5), 6))
        # ax.bar(model_rank['model'], model_rank['r2_test'], color='skyblue')
        # ax.set_xlabel('Model')
        # ax.set_ylabel('Best Test R2')
        # ax.set_title('Model Ranking by Best Test R2')
        # plt.xticks(rotation=45, ha='right')
        # fig.tight_layout()
        # plot_path = os.path.join(plots_dir, 'model_ranking_by_r2_test.pdf')
        # fig.savefig(plot_path, dpi=300)
        # plt.close(fig)
        # logger.info("Saved model ranking plot to %s", plot_path)
    except Exception as e:
        logger.warning("Could not generate model ranking CSV/plot: %s", e)

    # Generate markdown of best configuration per model
    try:
        best_per_model = df_results.loc[df_results.groupby('model')['r2_test'].idxmax()]
        md_lines = ["# Best Configurations per Model", ""]
        for idx_row, row in best_per_model.iterrows():
            model_name = row['model']
            params = row.get('params', {}) or {}
            md_lines.append(f"## {model_name}")
            md_lines.append("")
            # Experiment index
            md_lines.append(f"- **Experiment index**: {idx_row}")
            # Test metric
            r2_val = row.get('r2_test', None)
            if r2_val is not None:
                md_lines.append(f"- **Test R2**: {r2_val:.4f}")
                md_lines.append(f"- **Test MAE**: {row.get('mae_test', None):.4f}")
                md_lines.append(f"- **Test MAPE**: {row.get('mape_test', None):.4f}")
                md_lines.append(f"- **Test Pearson**: {row.get('pearson_test', None):.4f}")

            # Train metric
            r2_val = row.get('r2_train', None)
            if r2_val is not None:
                md_lines.append(f"- **Train R²**: {r2_val:.4f}")
                md_lines.append(f"- **Train MAE**: {row.get('mae_train', None):.4f}")
                md_lines.append(f"- **Train MAPE**: {row.get('mape_train', None):.4f}")
                md_lines.append(f"- **Train Pearson**: {row.get('pearson_train', None):.4f}")

            # Other metrics (optional)
            # Parameters block
            md_lines.append(f"- **Parameters**:")
            md_lines.append("```yaml")
            for k, v in sorted(params.items()):
                md_lines.append(f"{k}: {v}")
            md_lines.append("```")
            md_lines.append("")
        md_path = os.path.join(run_dir, "best_model_configs.md")
        with open(md_path, 'w') as md_file:
            md_file.write("\n".join(md_lines))
        logger.info("Saved best configs markdown to %s", md_path)
    except Exception as e:
        logger.warning("Could not generate best configs markdown: %s", e)
    # Generate summary boxplots for test metrics across models
    try:
        summary_dir = os.path.join(plots_dir, 'summary')
        make_dir(summary_dir)
        plot_summary_boxplots(df_results, summary_dir)
        logger.info("Saved summary boxplots to %s", summary_dir)
    except Exception as e:
        logger.warning("Could not generate summary boxplots: %s", e)

    # Plot only the best overall model's diagnostics
    try:
        # Identify best experiment by highest test R2
        best_idx, best_res = max(
            enumerate(results), key=lambda x: x[1].get('test', {}).get('r2', float('-inf'))
        )

        best_exp = experiments[best_idx]
        name = best_res['model']
        params = best_res.get('params', {}) or {}
        model = best_exp['model']
        # Prepare identifiers
        slug = _slugify_params(params)
        param_str = ", ".join([f"{k}={v}" for k, v in sorted(params.items())]) if params else "default"
        # Directory for best model plots
        best_plots_dir = os.path.join(plots_dir, 'best_model')
        make_dir(best_plots_dir)
        # Actual vs Predicted (test)
        data = best_res.get('test', {})
        y_true = data.get('y_true')
        y_pred = data.get('y_pred')

        if y_true is not None and y_pred is not None:
            fig, ax = plot_actual_vs_predicted(
                y_true, y_pred, model_name=name, split_name='test'
            )
            fig.supxlabel(f"Params: {param_str}", fontsize=8)
            fig.savefig(
                os.path.join(best_plots_dir, f"best_{name}_{slug}_test_actual_vs_predicted.png"), dpi=300
            )
            plt.close(fig)
        # Residuals histogram (test)
        if y_true is not None and y_pred is not None:
            fig, ax = plot_residuals(
                y_true, y_pred, model_name=name, split_name='test'
            )
            fig.supxlabel(f"Params: {param_str}", fontsize=8)
            fig.savefig(
                os.path.join(best_plots_dir, f"best_{name}_{slug}_test_residuals_histogram.png"), dpi=300
            )
            plt.close(fig)
        # Learning curve (train)
        try:
            fig = plot_learning_curve(
                model, X_train, y_train, title=f"{name} Learning Curve", cv=5, n_jobs=-1
            )
            fig.suptitle(f"Params: {param_str}" , fontsize=8)
            fig.savefig(
                os.path.join(best_plots_dir, f"best_{name}_{slug}_learning_curve.png"), dpi=300
            )
            plt.close(fig)
        except Exception as e:
            logger.warning("Could not generate learning curve for best model %s: %s", name, e)
        logger.info("Saved plots for best overall model to %s", best_plots_dir)
    except Exception as e:
        logger.warning("Could not generate plots for best overall model: %s", e)

    # Select and save best MLPRegressor configuration (compare only 1 vs 2 hidden layers)
    mlp_df = df_results[df_results['model'] == 'MLPRegressor']
    if not mlp_df.empty:
        best = select_best_model(mlp_df)
    else:
        best = None
    if best:
        logger.info("Best MLPRegressor configuration: %s", best)
        # Find corresponding model instance
        for exp, res in zip(experiments, results):
            if res.get("model") == best["model"] and res.get("params") == best["params"]:
                best_model = exp["model"]
                break
        else:
            best_model = None
        if best_model:
            # Compute variable significance for linear models
            try:
                if hasattr(best_model, "coef_"):
                    feat_names = X_train_df.columns.tolist()
                    df_signif = significance_test(best_model, X_train, y_train, feat_names)
                    signif_path = os.path.join(run_dir, "variable_significance.csv")
                    save_dataframe(df_signif, signif_path)
                    logger.info("Saved variable significance to %s", signif_path)
            except Exception as e:
                logger.warning("Could not compute variable significance: %s", e)

    logger.info("Pipeline finished. Outputs are in %s", run_dir)

