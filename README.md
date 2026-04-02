# Chest X-ray Grad-CAM Demo

This is a lightweight demo app you can host on AWS, another cloud VM, or in Docker.
It accepts a chest x-ray upload, runs a pretrained model, returns top predicted findings,
and displays a Grad-CAM heatmap overlay.

## What this is
- An educational demo
- A cloud-hostable radiology AI prototype
- A good starting point for investor demos, internal prototypes, or technical screens

## What this is not
- Not validated for patient care
- Not a medical device
- Not a clinical decision support product

## Model behavior
The app prefers `torchxrayvision` pretrained chest-xray weights.
If those weights are unavailable, it falls back to a torchvision model purely so the app can still run.
Fallback mode is infrastructure-only and should not be presented as medically meaningful.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:7860`

## Docker
```bash
docker build -t cxr-gradcam-demo .
docker run -p 7860:7860 cxr-gradcam-demo
```

## AWS options
### EC2
1. Launch a small GPU or CPU instance.
2. Install Docker.
3. Copy this project to the instance.
4. Build and run the container.
5. Open port 7860 in the security group.

### App Runner / ECS
- Build the Docker image
- Push it to ECR
- Deploy from ECR

## Recommended next upgrades
- Add DICOM upload support
- Replace Grad-CAM with segmentation overlays where possible
- Add sample demo images
- Log inference metrics
- Swap to FastAPI if you want a separate frontend

## Disclosure language for demos
Use language like:
> Educational AI demo using a pretrained chest x-ray model and Grad-CAM visualization.
> Not for clinical use.


## Streamlit version
```bash
streamlit run app_streamlit.py
```

This version provides a simple upload UI, predictions table, and Grad-CAM overlay in Streamlit.
