# COVID-19 Audio Detection Project

## Goal
Detect COVID-19 from audio data (breathing, cough, and speech) to classify subjects as COVID-positive or COVID-negative.

---

### 1. Data Preprocessing

1. **Data Augmentation**
   - **Goal**: Balance the dataset by increasing COVID-positive samples to match the number of COVID-negative samples.
   - **How**:
     - **Noise Injection**: Add colored noise to the audio using `AddColoredNoise` from `torch-audiomentations`.
     - **Pitch Shift**: Vary the pitch slightly using `PitchShift` from `torch-audiomentations`.
     - **Time Stretch**: Apply time stretching using `librosa.effects.time_stretch` to alter the speed of the audio.
   - **Process**: Each positive sample is augmented by generating 4–5 variations of all three types (`breathing`, `cough`, `speech`) until the total number of positive samples reaches the target count.
   - **Outcome**: An augmented dataset with a balanced number of COVID-positive and COVID-negative samples, enhancing diversity and robustness.

2. **Denoise Audio**
   - **Goal**: Improve audio clarity by reducing background noise, especially for augmented samples.
   - **How**: Apply spectral gating or a band-pass filter to remove unwanted frequencies using `scipy.signal`.
   - **Outcome**: Cleaner audio samples with enhanced feature quality, improving model performance.

3. **Segment Audio Files**
   - **Goal**: Standardize audio clips by segmenting them into uniform lengths suitable for model input.
   - **How**: Segment audio files into consistent 2–3 second clips using `librosa.effects.split`.
   - **Outcome**: Multiple shorter clips from each audio recording, providing standardized inputs for model training.

---

### 2. Feature Extraction

1. **Extract Time-Domain Features**
   - **Features**: Zero-crossing rate, RMS energy, and signal energy.
   - **How**: Use `librosa.feature` functions like `librosa.feature.zero_crossing_rate`.
   - **Outcome**: Feature vectors summarizing time-based audio patterns.

2. **Extract Frequency-Domain Features**
   - **Features**: Mel-Frequency Cepstral Coefficients (MFCCs).
   - **How**: Use `librosa.feature.mfcc` to extract 13 MFCCs for each sample.
   - **Outcome**: Frequency-based features useful for capturing audio characteristics.

3. **Create Spectrograms**
   - **Goal**: Convert audio into visual spectrograms for CNN models.
   - **How**: Use `librosa.feature.melspectrogram` to create Mel spectrograms, then save as images if needed.
   - **Outcome**: Image-based representations of audio for deep learning.

---

### 3. Model Selection and Training

- **Goal**: Train various models (SVM, Random Forest, CNN, ResNet) on extracted features to classify COVID status from audio data.
- **Outcome**: Trained models ready for evaluation.

---

### 4. Model Evaluation

- **Goal**: Assess model performance using metrics like accuracy, F1-score, and AUC-ROC.
- **Outcome**: Quantitative performance measures indicating model effectiveness.

---

## Project Setup

To set up the environment for this project, follow these steps.

### 1. Install Conda

If Conda is not already installed, download and install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### 2. Create the Conda Environment

Use the following command to create a new Conda environment for this project with all necessary dependencies:

```bash
conda create --name covid_audio_env python=3.8
```

### 3. Activate the Environment

Activate the newly created environment:

```bash
conda activate covid_audio_env
```

### 4. Activate the Environment

Install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
