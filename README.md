# Nano_Myo
Project Nano-Myo
10.3 KB Myoelectric Gesture Recognition for Low-Cost Prosthetics

Repo description: A TinyML pipeline for EMG-based gesture classification that fits in 10 KB — evaluated subject-independently on open-source data, targeting $3 ARM Cortex-M0+ microcontrollers.

A reproducible TinyML pipeline demonstrating that meaningful EMG gesture classification can run on a $3 microcontroller with no expensive hardware or proprietary signal processing required. The model is trained and evaluated under a strict subject-independent protocol on a publicly available dataset, making results directly comparable and reproducible by anyone.


Reproducing Results

Download Ninapro DB5 from Zenodo and place subject zips in MyDrive/Nano_Myo/
Run notebooks 01–06 in order on Google Colab
GPU runtime recommended for Notebook 04 (QAT training)

