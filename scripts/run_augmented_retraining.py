"""CLI script to execute Condition A (Real-Only) vs Condition B (Real + Synthetic) retraining."""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logger
from src.training.retrain_augmented_models import run_milestone5_experiments

logger = setup_logger("run_augmented_retraining_cli")


def main():
    """Execute Milestone 5 retraining experiments CLI."""
    logger.info("Initializing Milestone 5 retraining pipeline...")
    
    try:
        summary = run_milestone5_experiments()
        print("\n" + "=" * 80)
        print("MILESTONE 5: RETRAINING EXPERIMENT RESULTS (EVALUATION ON N=3,000 TEST SAMPLES)")
        print("=" * 80)
        eval3k = summary["evaluation_subset_3000"]
        
        print(f"{'Model & Condition':<38} | {'Accuracy':<10} | {'Macro F1':<10} | {'Brier':<10} | {'ECE':<10}")
        print("-" * 80)
        for model_name in ["Logistic_Regression", "Bidirectional_LSTM"]:
            data = eval3k[model_name]
            disp = model_name.replace("_", " ")
            print(f"{disp + ' (Real Only)':<38} | {data['Condition_A_Real_Only']['accuracy']:<10.4f} | {data['Condition_A_Real_Only']['macro_f1']:<10.4f} | {data['Condition_A_Real_Only']['brier_score']:<10.4f} | {data['Condition_A_Real_Only']['ece']:<10.4f}")
            print(f"{disp + ' (Real + 500 Synth)':<38} | {data['Condition_B_Real_Plus_100_Synthetic']['accuracy']:<10.4f} | {data['Condition_B_Real_Plus_100_Synthetic']['macro_f1']:<10.4f} | {data['Condition_B_Real_Plus_100_Synthetic']['brier_score']:<10.4f} | {data['Condition_B_Real_Plus_100_Synthetic']['ece']:<10.4f}")
            print(f"{'  -> Delta (% points)':<38} | {data['delta_accuracy_pct']:<+10.3f}% | {data['delta_macro_f1_pct']:<+10.3f}% | {'--':<10} | {'--':<10}")
            print(f"{'  -> Neutral Class F1 Delta':<38} | {data['delta_neutral_f1_pct']:<+10.3f}% | {'--':<10} | {'--':<10} | {'--':<10}")
            print("-" * 80)
        print("=" * 80 + "\n")
    except Exception as e:
        logger.error("Milestone 5 experiment failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

