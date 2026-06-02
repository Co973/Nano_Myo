# Nano_Myo
Project Nano-Myo
10.3 KB Myoelectric Gesture Recognition for Low-Cost Prosthetics

Repo description: A TinyML pipeline for EMG-based gesture classification that fits in 10 KB — evaluated subject-independently on open-source data, targeting $3 ARM Cortex-M0+ microcontrollers.

A reproducible TinyML pipeline demonstrating that meaningful EMG gesture classification can run on a $3 microcontroller with no expensive hardware or proprietary signal processing required. The model is trained and evaluated under a strict subject-independent protocol on a publicly available dataset, making results directly comparable and reproducible by anyone.

Where This Stands Out
Most published TinyML EMG classifiers benchmark under one or more favorable conditions that inflate reported accuracy: subject-dependent evaluation (training and testing on the same person), proprietary or lab-controlled datasets, or model sizes targeting 50 KB+ microcontrollers. This project takes the harder path on all three:
ConstraintThis ProjectEvaluation protocolSubject-independent (held-out subjects never seen in training)DatasetNinapro DB5 — public, freely available, widely benchmarkedModel size10.3 KB (INT8 .tflite)Target hardwareARM Cortex-M0+ @ 48 MHz (~$3)Inference time~0.62 ms (0.4% of latency budget)
Subject-independent evaluation is the relevant benchmark for any prosthetic that needs to work for more than one person. Training and testing on the same subject can inflate accuracy by 20–30 percentage points — a gap that disappears the moment the device is used by someone else.

Results
MetricValueModel size10.3 KB (.tflite, INT8)Target size≤ 16 KBStandard accuracy72.3%Functional accuracy80.6%Dangerous confusion rate14.1%Inference time (48 MHz)~0.62 msEstimated activation RAM186 bytesParameters7,594
Evaluated on 2 held-out subjects (S9, S10) never seen during training.
What Is Functional Accuracy?
Standard accuracy treats all misclassifications equally. Functional accuracy weights them by clinical consequence for a prosthetic user. Confusing "pinch" with "tripod grasp" during a picking task is survivable — the object still gets picked up. Accidentally firing an active gesture from rest is not.
Three cost tiers are defined:

Cost 1.0 — Rest misclassified as active gesture (phantom activation), or wrist flexion/extension swapped (moves in wrong direction)
Cost 0.6–0.8 — Opposite wrist rotations, active gesture classified as rest
Cost 0.2–0.4 — Similar grasp confusions (pinch / tripod / lateral)

Functional accuracy = 1 − (weighted misclassification cost / total predictions)
The 8-point gap between standard (72.3%) and functional (80.6%) accuracy indicates that the model's errors skew toward low-cost confusions. When it misclassifies, it tends to misclassify in forgivable ways.
Per-Class Accuracy
ClassAccuracyRest82.9%Lateral grasp61.2%Hand open49.9%Wrist extension35.1%Hand close31.4%Wrist supination27.9%Wrist pronation27.6%Tripod grasp27.5%Pinch29.0%Wrist flexion24.0%
Per-Subject Results
SubjectStandardFunctionalDangerous confusion rateS9 (held-out)71.7%78.0%17.7%S10 (held-out)72.8%83.1%10.4%

Hardware Target
SpecValueTarget MCUARM Cortex-M0+ @ 48 MHzModel size10,600 bytesActivation RAM186 bytesMACs per inference7,488Estimated inference time~0.62 msLatency budget150 ms (window size)
The model consumes 0.4% of the available latency budget per inference.

Note: Inference time and RAM figures are derived analytically from MAC counts and parameter sizes. No physical hardware was used in this project — see Known Limitations.


Architecture
Input: 80-dimensional DSP feature vector
↓
Dense(64, ReLU)
↓
Dense(32, ReLU)
↓
Dense(10, Softmax)

Total parameters: 7,594
Quantization: INT8 (QAT → PTQ → TFLite)
Feature Extraction
150 ms sliding window, 50% overlap, applied to 16-channel EMG at 200 Hz.
Five features extracted per channel (80 features total):

Mean Absolute Value (MAV)
Zero Crossings (ZC)
Root Mean Square (RMS)
Waveform Length (WL)
Slope Sign Changes (SSC)


Dataset
Ninapro DB5 — publicly available at zenodo.org/record/1000116

10 subjects, 16-channel EMG (2× Thalmic Myo armbands)
Exercise E2 only (17 gesture classes, 10 used here)
Train: subjects 1–8 | Test: subjects 9–10 (subject-independent)
Class balancing: Rest capped at 25,000 windows; active classes SMOTE'd to 5,000

Gesture Set (E2 Label Mapping)
ClassE2 LabelGesture00Rest11Hand open22Hand close34Pinch45Tripod grasp59Wrist flexion610Wrist extension711Wrist pronation812Wrist supination93Lateral grasp

Pipeline
01_data_loading.ipynb        Extract E2/E3 .mat files from subject zips
02_feature_extraction.ipynb  Windowing, 80-dim DSP features, subject split
03_baseline_training.ipynb   Float32 Dense baseline, architecture sweep
04_compression.ipynb         SMOTE → QAT → PTQ → .tflite
05_evaluation.ipynb          Standard + functional accuracy, confusion matrix
All notebooks run on Google Colab free tier. GPU is used for QAT training only.

Known Limitations
Wrist movements are the primary accuracy bottleneck. Flexion, extension, pronation, and supination average 27–35% accuracy under subject-independent evaluation. These gestures produce overlapping EMG patterns across individuals, and this overlap is especially pronounced in the Ninapro DB5 recordings, which were collected under controlled lab conditions without subject-specific calibration. The accuracy shortfall here reflects the difficulty of the task at this scale and evaluation protocol — not a claim about the ceiling of this architecture. Subject-specific fine-tuning or online calibration would substantially improve these classes, but would also compromise the subject-independent guarantee.
This project does not involve real hardware. Inference time (~0.62 ms) and RAM usage (186 bytes) are estimated from MAC counts and quantized parameter sizes. These figures are presented as theoretical bounds, not measured results. Physical validation on a Cortex-M0+ device is a natural next step.
Dangerous confusion rate of 14.1% (roughly 1 in 7 high-cost errors) is not acceptable for an unsupervised deployment. A confidence threshold or explicit rejection class would be necessary before any real-world use. Subject S9 (17.7%) drives this number disproportionately.
The primary contribution of this project is architectural and methodological — demonstrating what subject-independent EMG classification can achieve at extreme model sizes on open-source data — rather than a claim of deployment readiness.

Context
A myoelectric prosthetic hand typically requires a dedicated signal processing unit, a microcontroller with significant flash and RAM, and proprietary software. This project isolates the classification component of that pipeline and demonstrates that it can fit in 10 KB under a subject-independent evaluation protocol.
That is a narrow contribution. It does not solve prosthetics. It removes one specific barrier.

Reproducing Results

Download Ninapro DB5 from Zenodo and place subject zips in MyDrive/Nano_Myo/
Run notebooks 01–05 in order on Google Colab
GPU runtime recommended for Notebook 04 (QAT training)

Dependencies: TensorFlow 2.15, tensorflow-model-optimization, scikit-learn, imbalanced-learn, scipy, numpy, matplotlib
