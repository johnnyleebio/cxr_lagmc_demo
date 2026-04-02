import io
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
TOP_K = 5


@dataclass
class Prediction:
    label: str
    probability: float


# Default label set that matches common CXR demo tasks.
DEFAULT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fracture",
    "Infiltration",
    "Lung Lesion",
    "Lung Opacity",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
]


class CXRDemoModel:
    """
    Wrapper around a TorchXRayVision model with a torchvision fallback.

    Preferred path:
      - torchxrayvision DenseNet pretrained on chest x-rays
    Fallback path:
      - torchvision DenseNet121 ImageNet weights

    The fallback is only for infrastructure demos and should not be described
    as a clinically meaningful chest-x-ray model.
    """

    def __init__(self) -> None:
        self.model_type = None
        self.labels = DEFAULT_LABELS
        self.model = None
        self.target_module = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torchxrayvision as xrv  # type: ignore

            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            model = model.to(DEVICE).eval()
            self.labels = list(model.pathologies)
            self.model = model
            self.target_module = model.features.norm5
            self.model_type = "torchxrayvision"
        except Exception:
            from torchvision.models import DenseNet121_Weights, densenet121

            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            # Replace classifier head to output demo labels. The random head means
            # outputs are not medically meaningful unless the user swaps in a real
            # checkpoint. This keeps the app runnable while making the limitation
            # explicit.
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, len(self.labels))
            model = model.to(DEVICE).eval()
            self.model = model
            self.target_module = model.features.norm5
            self.model_type = "torchvision_fallback"

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")
        arr = np.array(image).astype(np.float32)
        arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        arr = arr / 255.0

        # Chest x-ray models generally expect single-channel input. DenseNet in
        # TorchXRayVision supports this. The fallback model was adapted to accept
        # 3-channel input by replication at inference time.
        if self.model_type == "torchxrayvision":
            tensor = torch.from_numpy(arr)[None, None, ...]
        else:
            arr3 = np.stack([arr, arr, arr], axis=0)
            tensor = torch.from_numpy(arr3)[None, ...]

        return tensor.to(DEVICE)

    def predict_with_gradcam(self, image: Image.Image) -> Tuple[List[Prediction], np.ndarray, str]:
        tensor = self._preprocess(image)
        activations = []
        gradients = []

        def fwd_hook(_module, _inp, out):
            activations.append(out)

        def bwd_hook(_module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_fwd = self.target_module.register_forward_hook(fwd_hook)
        handle_bwd = self.target_module.register_full_backward_hook(bwd_hook)

        try:
            logits = self.model(tensor)
            if isinstance(logits, dict):
                logits = logits["logits"]

            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
            top_idx = np.argsort(probs)[::-1][:TOP_K]
            predictions = [Prediction(self.labels[i], float(probs[i])) for i in top_idx]

            target_index = int(top_idx[0])
            self.model.zero_grad(set_to_none=True)
            logits[0, target_index].backward()

            cam = self._build_gradcam(activations[0], gradients[0], image)
            note = self._model_note()
            return predictions, cam, note
        finally:
            handle_fwd.remove()
            handle_bwd.remove()

    def _build_gradcam(self, activation: torch.Tensor, gradient: torch.Tensor, image: Image.Image) -> np.ndarray:
        # activation/gradient shape: [1, C, H, W]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        base = image.convert("RGB")
        base_arr = np.array(base)
        h, w = base_arr.shape[:2]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(base_arr, 0.6, heatmap, 0.4, 0)
        return overlay

    def _model_note(self) -> str:
        if self.model_type == "torchxrayvision":
            return (
                "Using TorchXRayVision pretrained CXR weights. This is an educational "
                "demo only. The heatmap is a Grad-CAM visualization, not a diagnosis."
            )
        return (
            "Using fallback torchvision weights because TorchXRayVision was not available. "
            "Predictions in fallback mode are not medically meaningful. Install the full "
            "requirements on your server before showing this demo."
        )


MODEL = CXRDemoModel()


def predict(image: Image.Image):
    if image is None:
        raise gr.Error("Please upload a chest x-ray image.")

    preds, overlay, note = MODEL.predict_with_gradcam(image)
    table = [[p.label, f"{p.probability:.3f}"] for p in preds]
    return overlay, table, note


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Chest X-ray AI Demo") as demo:
        gr.Markdown(
            """
            # Chest X-ray AI Demo
            Upload a chest x-ray and the app will return top predicted findings plus a Grad-CAM heatmap.

            **Important:** This is an educational demo. Do not use it for patient care.
            """
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload chest x-ray")
                run_btn = gr.Button("Run inference")
            with gr.Column():
                output_image = gr.Image(label="Grad-CAM overlay")
                output_table = gr.Dataframe(headers=["Finding", "Probability"], row_count=TOP_K, col_count=2)
                output_note = gr.Textbox(label="Model note")

        run_btn.click(predict, inputs=input_image, outputs=[output_image, output_table, output_note])

        gr.Markdown(
            """
            ### Hosting
            - Local: `python app.py`
            - Docker: see the included `Dockerfile`
            - AWS: deploy the container to EC2, ECS, or App Runner
            """
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    app.launch(server_name=host, server_port=port, show_error=True)
