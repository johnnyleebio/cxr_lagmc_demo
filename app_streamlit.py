import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
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
    """Pretrained CXR model wrapper with Grad-CAM support.

    Preferred path:
      - TorchXRayVision DenseNet pretrained on chest x-rays
    Fallback path:
      - torchvision DenseNet121 ImageNet weights

    The fallback keeps the demo runnable, but it is not medically meaningful.
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
            for module in model.modules():
                if isinstance(module, torch.nn.ReLU):
                    module.inplace = False
            model = model.to(DEVICE).eval()
            self.labels = list(model.pathologies)
            self.model = model
            self.target_module = model.features.norm5
            self.model_type = "torchxrayvision"
        except Exception:
            from torchvision.models import DenseNet121_Weights, densenet121

            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
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
        out.register_hook(lambda grad: gradients.append(grad))

    handle_fwd = self.target_module.register_forward_hook(fwd_hook)

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

    def _build_gradcam(self, activation: torch.Tensor, gradient: torch.Tensor, image: Image.Image) -> np.ndarray:
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


@st.cache_resource(show_spinner=False)
def load_model() -> CXRDemoModel:
    return CXRDemoModel()


st.set_page_config(page_title="Chest X-ray AI Demo", layout="wide")
st.title("Chest X-ray AI Demo")
st.caption("Upload a chest x-ray to generate top predicted findings and a Grad-CAM heatmap.")
st.warning("Educational demo only. Not for clinical use.")

model = load_model()

with st.sidebar:
    st.subheader("Deployment notes")
    st.write(f"Device: `{DEVICE}`")
    st.write(f"Model path: `{model.model_type}`")
    st.markdown(
        """
        **Run locally**
        ```bash
        streamlit run app_streamlit.py
        ```
        """
    )

uploaded = st.file_uploader("Upload chest x-ray", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with st.spinner("Running inference..."):
        preds, overlay, note = model.predict_with_gradcam(image)

    with col2:
        st.subheader("Grad-CAM overlay")
        st.image(overlay, use_container_width=True)

    st.subheader("Top findings")
    st.table(
        {
            "Finding": [p.label for p in preds],
            "Probability": [f"{p.probability:.3f}" for p in preds],
        }
    )

    st.info(note)
else:
    st.markdown(
        """
        ### What this demo shows
        - top predicted chest x-ray findings
        - a Grad-CAM saliency overlay
        - a simple cloud-hostable workflow for demos
        """
    )
