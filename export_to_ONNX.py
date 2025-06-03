from pathlib import Path
from optimum.exporters.onnx import main_export

# Export MPNet model
main_export(
    model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    output=Path("onnx/mpnet"),
    task="feature-extraction"
)

# Export RoBERTa emotion model for tone
main_export(
    model_name_or_path="cardiffnlp/twitter-roberta-base-emotion",
    output=Path("onnx/emotion"),
    task="feature-extraction"
)