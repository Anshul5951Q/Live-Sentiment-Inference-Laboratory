"""Test graceful degradation when model artifacts are completely absent."""
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

with tempfile.TemporaryDirectory() as tmpdir:
    from config.settings import settings
    original_models_dir = settings.models_dir
    settings.models_dir = Path(tmpdir)
    
    from ui.pages.page_sentiment_lab import (
        check_lr_available,
        check_bilstm_available,
        check_bert_available,
    )
    
    lr_ok, lr_msg = check_lr_available()
    bilstm_ok, bilstm_msg = check_bilstm_available()
    bert_ok, bert_msg = check_bert_available()
    
    print("Testing behavior with ZERO model artifacts present:")
    print(f"LR: available={lr_ok}, msg='{lr_msg}'")
    print(f"BiLSTM: available={bilstm_ok}, msg='{bilstm_msg}'")
    print(f"BERT: available={bert_ok}, msg='{bert_msg}'")
    
    assert not lr_ok, "LR should report unavailable when artifacts missing"
    assert not bilstm_ok, "BiLSTM should report unavailable when artifacts missing"
    assert not bert_ok, "BERT should report unavailable when artifacts missing"
    
    settings.models_dir = original_models_dir
    print("✓ Verification successful: All model paths degrade gracefully with clear status messages when artifacts are missing!")

