# Automotive Anomaly Detection

## Project Overview
Automated instance segmentation to detect and highlight damages (scratches, dents) on car images using YOLOv8. This project aims to replace manual car inspection with a fast, objective, and scalable computer vision solution.

## System Design

### Dataset Requirements
- Images of cars with annotated damage masks
- Source: Roboflow, custom collection, or open datasets

### Architecture
1. **Pre-processing:** Resize, normalize, augment images
2. **Model:** YOLOv8 instance segmentation (`yolov8n-seg.pt` or custom-trained)
3. **Training Pipeline:**
   - Data loading
   - Model training (with validation split)
   - Metrics logging
4. **Post-processing:**
   - Overlay masks on images
   - Generate summary metrics

### Challenges
- Data imbalance (few damaged vs. many undamaged images)
- Varying lighting/angles
- Small or subtle damages

### Performance Metrics
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall
- Inference latency (for real-time use)

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and validate the model
Set your Roboflow API key as an environment variable, then run the training script:
```bash
export ROBOFLOW_API_KEY=your_api_key_here   # on Windows (PowerShell): $env:ROBOFLOW_API_KEY = "your_api_key_here"
python model.py
```

### 3. Run the Streamlit demo app
```bash
streamlit run app.py
```

---

## Project Stages

1. **Proposal:** See `proposal_template.md` for the initial project proposal.
2. **System Design:** See above and `report_template.md` for detailed design.
3. **Implementation:** See `model.py` and `app.py`.
4. **Reporting:** Use `report_template.md` for your final report.

---

## Authors
- Ali Hassan 
