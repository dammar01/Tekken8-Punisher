# Tekken8 Punisher

**Status:** Project is currently under development 🚧

## Overview
**Tekken8 Punisher** is a specialized tool designed to analyze and extract frame data from Tekken 8 replays. The application utilizes **YOLO-based object detection** and **OCR techniques** to recognize player inputs, hit statuses, and other key gameplay information. By leveraging advanced **computer vision** and **machine learning**, this tool aims to provide precise insights into how moves interact and how players can optimize their punishes.

## Features
- 🎯 **Input Detection** – Recognizes player input notations using YOLO-based object detection.
- 🏆 **Hit Analysis** – Determines whether an attack hits, is blocked, or dodged.
- 📊 **Frame Data Extraction** – Captures key frame information like startup, advantage, and attack properties.
- 🔍 **OCR for Game Elements** – Extracts critical gameplay data, including health, damage, and time.
- 🎥 **Replay Analysis** – Processes replay footage frame by frame for precise data logging.

## How It Works
1. **Detect Inputs:** Uses YOLOv8 to analyze player inputs directly from the replay footage.
2. **Extract Frame Data:** Leverages OCR and detection models to extract relevant frame and game status information.
3. **Analyze Interactions:** Identifies punishes, sidesteps, and frame advantages to provide actionable insights.
4. **Generate Data:** Compiles extracted data into structured tables for further analysis.

## Requirements
- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- Tesseract OCR
- NumPy & Pandas

## Notes
🔹 This project is **actively in development**, meaning features may change or improve over time.  
🔹 Contributions and feedback are welcome, but the tool is currently tailored for **personal research and training purposes**.  
🔹 Model training and dataset augmentation are being optimized to improve detection accuracy.  

---

### **Disclaimer**
This project is not affiliated with or endorsed by **Bandai Namco Entertainment** or the **Tekken** franchise. It is an independent project for research and educational purposes.

