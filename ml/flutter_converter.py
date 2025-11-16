# save as: flutter_converter.py  (run from ml/ directory)

from pathlib import Path
import torch
from train import CalorieRegressor

DEVICE = "cpu"

ML_ROOT = Path(__file__).parent.resolve()
ARTIFACTS_DIR = ML_ROOT / "artifacts"
WEIGHTS_PATH = ARTIFACTS_DIR / "calorie_regressor_efficientnet_b0.pt"

# export directly into Flutter project:
FLUTTER_ONNX_PATH = (
    ML_ROOT.parent  # .../ThinK/
    / "calorie_counting"
    / "assets"
    / "models"
    / "calorie_regressor_efficientnet_b0.onnx"
)


def main():
    # 1) rebuild model + load weights
    model = CalorieRegressor()
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    # 2) dummy input (3x224x224)
    dummy = torch.randn(1, 3, 224, 224, device=DEVICE)

    # 3) ensure output dir in Flutter project exists
    FLUTTER_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 4) export ONNX directly into calorie_counting/assets/models/
    torch.onnx.export(
        model,
        dummy,
        FLUTTER_ONNX_PATH.as_posix(),
        input_names=["input"],
        output_names=["log_kcal"],  # log(calories + 1)
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch_size"},
            "log_kcal": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    print(f"[OK] ONNX model saved to: {FLUTTER_ONNX_PATH}")


if __name__ == "__main__":
    main()
