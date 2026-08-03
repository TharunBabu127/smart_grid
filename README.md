# Smart Grid Fault Detection

This repository contains the source code for the **Smart Grid Fault Detection System**, which classifies electrical power line faults using RF signal processing and Deep Learning.

## Getting Started

The main source code and its dedicated documentation are located in the `SmartGrid/` directory.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TharunBabu127/Smart_Grid-new-.git
   cd "Smart_Grid-new-/SmartGrid"
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   * **Windows:**
     ```cmd
     venv\Scripts\activate
     ```
   * **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

Make sure you are in the `SmartGrid` directory when running these commands. You can execute the steps in the following sequence depending on what you want to do:

1. **Generate IQ Signals:** `python src/generate_iq.py`
2. **Generate FFT:** `python src/fft_processing.py`
3. **Generate Spectrograms:** `python src/spectrogram.py`
4. **Train CNN Model:** `python src/train.py`
5. **Evaluate Model:** `python src/evaluate.py`
6. **Predict:** `python src/predict.py`
7. **Real-time Simulation:** `python src/realtime.py`

For more detailed information on the project architecture, features, and model structure, please refer to the comprehensive [SmartGrid/README.md](SmartGrid/README.md) file.
