import os
from datetime import datetime

from config.parser import load_model_configs
from data.loader import load_raw_data, clean_data
from data.preprocess import preprocess
from training.trainer import train_and_evaluate
from training.evaluation import results_to_dataframe, select_best_model
from utils.io import make_dir, save_dataframe, save_model
from utils.logging import get_logger

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

    # Train and evaluate models
    results = []
    for exp in experiments:
        name = exp["name"]
        params = exp["params"]
        model = exp["model"]
        logger.info("Training model %s with params %s", name, params)
        res = train_and_evaluate(model, name, params, X_train, y_train, X_test, y_test)
        results.append(res)

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