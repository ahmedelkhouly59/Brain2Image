# 🧠 Brain2Image: Converting EEG Signals into Images

## Overview

This research project, developed as part of our PPL (Project-Based Learning) initiative, focuses on converting EEG signals into corresponding images using deep learning. We began with the implementation of the **Brain2Image** paper, and progressively enhanced the methodology with our contributions. The final system is deployed as a web application using the **Django** framework.

---

## 🔬 Research Objectives

- Replicate the architecture proposed in the "Brain2Image: Converting Brain Signals into Images" paper.
- Enhance signal processing and image reconstruction with our custom techniques.
- Deploy the solution on a web platform for interactive usage.

---

## 🧪 Methodology

### 1. Baseline Implementation
We implemented the original Brain2Image approach which involves:
- EEG signal preprocessing
- Feature extraction from EEG using an encoder
- Image generation using a decoder or generative model

### 2. Our Contributions
- Improved EEG preprocessing pipeline
- Custom regularization in latent space for better feature learning
- Enhanced decoder using Conditional GAN for more realistic outputs
- Integration of multiple EEG datasets for better generalization

### 3. Deployment
- Built using Django framework
- Upload EEG files and generate images via browser interface
- Simple and responsive UI for ease of use


