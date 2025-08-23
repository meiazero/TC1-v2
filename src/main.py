import argparse
from pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline for real estate valuation")
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
        "--test-size", type=float, default=1/3, help="Proportion of data for testing"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--remove-outliers", action="store_true", default=False, help="Remove outliers from data before processing"
    )

    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state,
        remove_outliers=args.remove_outliers
    )

if __name__ == "__main__":
    main()
