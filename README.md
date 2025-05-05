# Real-Time Leaf Disease Detection using Color Segmentation and Morphological Operations

## Team Members
- **S Ananya** - 4SO22CD042
- **Sahana Rao** - 4SO22CD043
- **Tharakan Vidya Ravunni** - 4SO22CD059

---

## Problem Statement

In agriculture, timely detection of diseases in leaves is crucial to prevent crop loss and improve yield. This project provides a solution to detect diseases in leaves by analyzing images or live video feeds from a webcam. The system:
- Detects potential diseases in leaves based on color segmentation.
- Provides a visual representation of affected areas.
- Calculates the percentage of the leaf affected by disease.

---

## Techniques Used

- **Color Segmentation (HSV)**: Detects areas of the leaf with color patterns indicating disease.
- **Morphological Operations**: Used to clean up the detected regions for more accurate results (using `open` and `close` operations).
- **Image Analysis**: Processes leaf images to highlight the affected regions and compute the percentage of the leaf that is diseased.
- **Real-Time Webcam Feed**: Uses OpenCV to process live video streams and detect diseases in real-time.

---

## Program Flow

1. **Select Mode**: Choose between analyzing a static leaf image or running real-time detection using a webcam.
2. **Leaf Image Analysis**:
   - Load and preprocess the leaf image.
   - Apply color segmentation in HSV to detect potential diseased areas.
   - Clean the segmented mask using morphological operations.
   - Display results showing the percentage of the leaf affected by disease.
3. **Real-Time Webcam Detection**:
   - Capture video feed from the webcam.
   - Detect diseased regions in real-time.
   - Highlight diseased areas with a red overlay.
   - Display the real-time feed with marked affected regions.
4. **Exit the Program**: Press 'q' or close the window to exit the real-time detection mode.

---

## Evaluation Metrics

| Metric                       | Value                             |
|-----------------------------|------------------------------------|
| Disease Detection Method    | Color Segmentation (HSV)           |
| Morphological Operations    | Open and Close for cleaning        |
| Real-Time FPS               | ~20-30 FPS on standard webcam      |
| Affected Area Displayed     | Yes (percentage of affected pixels) |

---

## 🖼Sample Results

---
![Screenshot 2025-05-05 141627](https://github.com/user-attachments/assets/fe205c66-99ee-41fa-aac6-d7f69c4695db)

![Screenshot 2025-05-05 141026](https://github.com/user-attachments/assets/936fcf88-9bf8-412c-af1e-c8b60a07d210)

