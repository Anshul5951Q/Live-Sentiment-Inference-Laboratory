"""CLI script to run comprehensive Responsible AI, Bias, and Safety Audits (Unit 5)."""
import json
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logger
from src.responsible_ai.bias_analyzer import RegionalBiasAuditor
from src.responsible_ai.hallucination_evaluator import HallucinationEvaluator
from src.responsible_ai.safety_guardrails import SafetyGuardrails
from src.models.predictor import SentimentPredictor
from src.models.lstm_classifier import KerasBiLSTMClassifier
from src.models.text_tokenizer import TextTokenizer
from src.llm.gemini_provider import GeminiProvider

logger = setup_logger("run_responsible_ai_audit_cli")


def main():
    """Execute complete Unit 5 Responsible AI audit suite."""
    logger.info("Initializing Unit 5 Responsible AI audit suite...")
    
    out_dir = settings.results_dir / "responsible_ai"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Counterfactual Regional Bias Audit
    bias_auditor = RegionalBiasAuditor(output_dir=out_dir)
    
    # A. Audit Logistic Regression
    lr_dir = settings.models_dir / "logistic_regression"
    lr_pred = SentimentPredictor.from_artifacts(lr_dir)
    def lr_predict_fn(texts):
        results = lr_pred.predict(texts)
        preds = [r["predicted_label"] for r in results]
        confs = [r["confidence"] for r in results]
        return preds, confs
        
    lr_bias_results = bias_auditor.audit_model("Logistic_Regression", lr_predict_fn)
    
    # B. Audit BiLSTM
    lstm_dir = settings.models_dir / "lstm"
    lstm_clf = KerasBiLSTMClassifier().load(lstm_dir / "lstm_model.keras")
    lstm_tok = TextTokenizer().load(lstm_dir / "tokenizer.json")
    def lstm_predict_fn(texts):
        seqs = lstm_tok.transform(texts)
        probs = lstm_clf.predict_proba(seqs)
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)
        return preds, confs
        
    lstm_bias_results = bias_auditor.audit_model("Bidirectional_LSTM", lstm_predict_fn)
    
    # 2. Hallucination & Groundedness Audit on Gemini
    logger.info("Running Hallucination & Groundedness audit on Google Gemini...")
    provider = GeminiProvider()
    halluc_eval = HallucinationEvaluator(output_dir=out_dir)
    halluc_results = halluc_eval.evaluate_provider(provider, model_name="Google Gemini (gemini-3.5-flash-lite)")
    
    # 3. Enterprise Safety Guardrails Audit
    logger.info("Running Enterprise Safety Guardrails test suite...")
    guardrails = SafetyGuardrails()
    safety_results = guardrails.run_guardrail_test_suite()
    with open(out_dir / "safety_guardrails_report.json", "w") as f:
        json.dump(safety_results, f, indent=2)
        
    # 4. Print Summary Report
    print("\n" + "=" * 80)
    print("UNIT 5: RESPONSIBLE AI, BIAS & SAFETY AUDIT SUMMARY")
    print("=" * 80)
    print(f"Counterfactual Test Pairs:         {lr_bias_results['num_counterfactual_pairs']} templates x {lr_bias_results['num_regional_groups']} cultural groups ({lr_bias_results['total_test_samples']} reviews)")
    print(f"Logistic Regression Parity Ratio:  {lr_bias_results['counterfactual_parity_ratio']*100:.1f}% (Max Group Disparity: {lr_bias_results['max_accuracy_disparity']*100:.1f}%)")
    print(f"BiLSTM Parity Ratio:               {lstm_bias_results['counterfactual_parity_ratio']*100:.1f}% (Max Group Disparity: {lstm_bias_results['max_accuracy_disparity']*100:.1f}%)")
    print("-" * 80)
    print(f"Gemini Groundedness Score:         {halluc_results['groundedness_rate']*100:.1f}%")
    print(f"Unsupported Claim (Halluc.) Rate:  {halluc_results['unsupported_claim_rate']*100:.1f}%")
    print(f"Entity Recall from Reviews:        {halluc_results['mean_grounded_entity_recall']*100:.1f}%")
    print("-" * 80)
    print(f"Safety Guardrails Pass Rate:       {safety_results['pass_rate']*100:.1f}% ({safety_results['total_tests']} attack/boundary vectors tested)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
