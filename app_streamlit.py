from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
TOP_K = 5


@dataclass
class Prediction:
    label: str
    probability: float


class CXRDemoModel:
    """TorchXRayVision-based CXR demo with Grad-CAM.

    Notes:
    - Uses TorchXRayVision preprocessing as documented by the project.
    - Educational demo only. Not for clinical use.
    """

    def __init__(self) -> None:
        self.model = None
        self.labels: List[str] = []
        self.target_module = None
        self.transform = None
        self._load_model()

    def _load_model(self) -> None:
        import torchxrayvision as xrv  # type: ignore

        model = xrv.models.DenseNet(weights="densenet121-res224-all")

        # Avoid in-place ReLU issues with hooks / gradients.
        for module in model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False

        model = model.to(DEVICE).eval()

        self.model = model
        self.labels = list(model.pathologies)

        # Use a spatially richer layer than the final norm for better Grad-CAM.
        # denseblock4 is usually a better visualization target than the final norm.
        self.target_module = model.features.denseblock4

        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(IMAGE_SIZE),
        ])

    def _crop_borders(self, image: Image.Image) -> Image.Image:
        """Crop borders / markers to reduce shortcut attention."""
        arr = np.array(image)

        if arr.ndim == 2:
            h, w = arr.shape
        else:
            h, w = arr.shape[:2]

        # More from top, modest side crop, less from bottom
        y0 = int(0.15 * h)
        y1 = int(0.95 * h)
        x0 = int(0.05 * w)
        x1 = int(0.95 * w)

        arr = arr[y0:y1, x0:x1]
        return Image.fromarray(arr)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        import torchxrayvision as xrv  # type: ignore

        arr = np.array(image)

        # TorchXRayVision examples use normalize(img, 255), then single channel,
        # then XRayCenterCrop + XRayResizer(224).  [oai_citation:1‡GitHub](https://github.com/mlmed/torchxrayvision/blob/main/docs/source/index.rst?utm_source=chatgpt.com)
        arr = xrv.datasets.normalize(arr, 255)

        if arr.ndim == 3:
            arr = arr.mean(2)

        arr = arr[None, ...]  # (1, H, W)
        arr = self.transform(arr)
        tensor = torch.from_numpy(arr).unsqueeze(0).float().to(DEVICE)  # (1, 1, H, W)
        return tensor

    def predict_with_gradcam(self, image: Image.Image) -> Tuple[List[Prediction], np.ndarray, str]:
        image = self._crop_borders(image)
        tensor = self._preprocess(image)

        self.model.zero_grad(set_to_none=True)

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))

        handle = self.target_module.register_forward_hook(forward_hook)

        try:
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

            top_idx = np.argsort(probs)[::-1][:TOP_K]
            predictions = [Prediction(self.labels[i], float(probs[i])) for i in top_idx]

            target_index = int(top_idx[0])
            logits[0, target_index].backward()

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            cam = self._build_gradcam(activations[-1], gradients[-1], image)
            note = self._model_note(predictions[0].label)
            return predictions, cam, note

        finally:
            handle.remove()

    def _build_gradcam(
        self,
        activation: torch.Tensor,
        gradient: torch.Tensor,
        image: Image.Image,
    ) -> np.ndarray:
        # activation, gradient: (1, C, H, W)
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        base = image.convert("RGB")
        base_arr = np.array(base)
        h, w = base_arr.shape[:2]

        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(base_arr, 0.8, heatmap, 0.2, 0)
        return overlay

    def _model_note(self, top_label: str) -> str:
        return (
            f"Using TorchXRayVision pretrained CXR weights. Top Grad-CAM target: {top_label}. "
            "This is an educational visualization only, not a diagnosis."
        )


@st.cache_resource(show_spinner=False)
def load_model() -> CXRDemoModel:
    return CXRDemoModel()


st.set_page_config(page_title="Chest XR AI Demo - LA General / USC", layout="wide")
st.title("Chest XR AI Demo - LA General / USC")
st.caption("Upload a chest X-ray to generate top predicted findings and a Grad-CAM heatmap.")
st.warning("Educational demo only. Not for clinical use.")

model = load_model()

with st.sidebar:
    st.subheader("Deployment notes")
    st.write(f"Device: `{DEVICE}`")
    st.write("Model path: `torchxrayvision`")
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
