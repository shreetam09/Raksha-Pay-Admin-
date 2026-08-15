import os
import pandas as pd
import numpy as np
from ml.train_isolation_forest import main as train_if_main
import ml.train_isolation_forest

def test_train_isolation_forest(tmp_path, monkeypatch):
    # Setup temporary directories
    processed_dir = tmp_path / "processed"
    artifacts_dir = processed_dir / "artifacts"
    processed_dir.mkdir()
    artifacts_dir.mkdir()
    
    # Override paths in the module
    monkeypatch.setattr(ml.train_isolation_forest, "PROCESSED_DIR", str(processed_dir))
    monkeypatch.setattr(ml.train_isolation_forest, "ARTIFACTS_DIR", str(artifacts_dir))
    
    # Create dummy data
    np.random.seed(42)
    X_train_normal = pd.DataFrame({
        "feat1": np.random.rand(100),
        "feat2": np.random.rand(100)
    })
    
    X_full = pd.DataFrame({
        "feat1": np.random.rand(200),
        "feat2": np.random.rand(200)
    })
    
    meta_full = pd.DataFrame({
        # Ensure at least two classes per source for AUC computation
        "label": np.array([0]*50 + [1]*50 + [0]*50 + [1]*50),
        "source": ["ulb"] * 100 + ["paysim"] * 100
    })
    
    # Save dummy data
    X_train_normal.to_parquet(processed_dir / "X_train_normal.parquet", index=False)
    X_full.to_parquet(processed_dir / "X_full.parquet", index=False)
    meta_full.to_parquet(processed_dir / "meta_full.parquet", index=False)
    
    # Run the training script
    train_if_main()
    
    # Verify outputs
    assert (artifacts_dir / "isolation_forest.joblib").exists()
    assert (processed_dir / "if_scores.parquet").exists()
    
    # Verify the scores dataframe
    scores = pd.read_parquet(processed_dir / "if_scores.parquet")
    assert "if_anomaly_score" in scores.columns
    assert "if_pred_label" in scores.columns
    assert len(scores) == 200
