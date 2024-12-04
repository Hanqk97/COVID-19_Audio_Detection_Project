# COVID-19 Audio Detection Project

## Goal
This project aims to detect COVID-19 infection from audio recordings (breathing, cough, and speech) by classifying subjects as COVID-positive or COVID-negative using various machine learning models.

---
## Installation and Setup

### Step 1: Install Conda
Download and install Conda from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### Step 2: Create a Conda Environment
Use the following command to create a new Conda environment:

```bash
conda create --name covid_audio_env python=3.8
```
### Step 3: Activate the Environment
Activate the newly created environment:

```bash
conda activate covid_audio_env
```

### Step 4: Install Dependencies
Install all required dependencies:
```bash
pip install -r requirements.txt
```

---

## Workflow

### Data Preprocessing and Feature Extraction
The preprocessing steps transform raw audio into feature representations suitable for machine learning models. The following scripts execute this workflow sequentially:

1. **Reclassify Data**:
   - Script: `1_raw_data_reclassified.py`
   - **Purpose**: Organize and filter raw audio data into the required classes.
   - **Output**: Reclassified audio data saved in the `data` folder.

2. **Segment Audio**:
   - Script: `2_segment.py`
   - **Purpose**: Segment audio recordings into fixed durations to standardize input size.
   - **Output**: Segmented audio files saved in the `data` folder.

3. **Augment Data**:
   - Script: `3_augmentation.py`
   - **Purpose**: Apply noise injection, pitch shifting, and time stretching to balance class distributions.
   - **Output**: Augmented audio data stored alongside the original files.

4. **Extract Features**:
   - Script: `4_feature_extract.py`
   - **Purpose**: Extract time-domain and frequency-domain features from audio data.
   - Features include:
     - **MFCCs** (Mel-Frequency Cepstral Coefficients).
     - **Spectral Centroid**.
     - **Log-Mel Spectrograms**.
     - **Chroma Features**.
     - **RMS Energy**.
   - **Output**: Feature representations saved as JSON files in `data/extracted_features`.

---

### Model Training and Evaluation

The following machine learning models are trained and evaluated on the extracted features. Each model script includes implementations for:
- Training and saving the best-performing model based on F1 score.
- Testing saved models on unseen validation datasets.

1. **Convolutional Neural Network (CNN)**:
   - Scripts: `CNN.py` then  `CNN_raw.py`
   - **Description**: Trains and evaluates a CNN for binary classification using spectrogram-based inputs.

2. **ResNet**:
   - Scripts: `Resnet.py` then `Resnet_raw.py`
   - **Description**: Adapts a ResNet-18 architecture for classifying audio-based features.

3. **Support Vector Machine (SVM)**:
   - Scripts: `SVM.py` then `SVM_raw.py`
   - **Description**: Implements a pipeline with feature standardization and SVM for binary classification.

4. **Random Forest**:
   - Scripts: `RandomForest.py` then `RandomForest_raw.py`
   - **Description**: Uses a Random Forest model for classification and feature importance analysis.

5. **XGBoost**:
   - Scripts: `XGBoost.py` then `XGBoost_raw.py`
   - **Description**: Employs an XGBoost classifier for robust performance on audio feature data.

---

## Outputs and Results
- Each model's results, including confusion matrices and evaluation metrics (F1 score, precision, recall, accuracy, AUC), are saved in the `result` directory.
- Saved models:
  - **CNN**: `CNN.pth`
  - **ResNet**: `ResNet.pth`
  - **SVM**: `SVM.joblib`
  - **Random Forest**: `RandomForest.joblib`
  - **XGBoost**: `XGBoost.json`
- Outputs include timestamped directories to ensure reproducibility and tracking of results.


---

## Visualization

### Grouped Bar Charts
- Script: `grounped_bar_chart.py`
- **Description**: Plots grouped bar charts comparing F1 scores and AUC values across folds for raw and augmented data for each model.
- **Output**: Charts saved in the `Plot` directory.

### Heatmaps
- Script: `heatmap.py`
- **Description**: Creates heatmaps showing averaged F1 scores and AUC values for each model across three experiment types:
  - RTRV: Raw Training and Raw Validation
  - ATAV: Augmented Training and Augmented Validation
  - ATRV: Augmented Training and Raw Validation
- **Output**: Heatmaps saved in the `Plot` directory.