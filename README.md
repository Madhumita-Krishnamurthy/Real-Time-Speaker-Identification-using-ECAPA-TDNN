# Real-Time-Speaker-Identification-using-ECAPA-TDNN
This project implements a **real-time speaker identification system** using the **ECAPA-TDNN** model from [SpeechBrain](https://speechbrain.github.io/).   It enables users to speak into a microphone through a web interface, processes the audio in real time, and identifies the speaker based on pre-enrolled voice embeddings. (Note: Some files such as models and data are not included in this repository.)

## Project Overview

The system records audio via the browser, sends it to a Flask backend, and performs **speaker identification** using a pretrained ECAPA-TDNN model.  
It includes an interactive web interface with countdown, recording feedback, and live prediction results.

## Key Features

- 🎧 **Real-Time Recording:** Capture audio directly from the browser (using MediaRecorder API)  
- 🧠 **Deep Learning Model:** Uses the pretrained ECAPA-TDNN from SpeechBrain  
- 🧾 **Voice Enrollment:** Stores speaker embeddings for future identification  
- ⚡ **Fast Inference:** Identifies speaker after a short recording  
- 💻 **Flask Web App:** Clean interface with purple-white theme, countdown, and retry options  

## Tech Stack

### Backend
- **Flask** – Backend web framework  
- **SpeechBrain (ECAPA-TDNN)** – Speaker embedding and recognition  
- **PyTorch** – Deep learning engine  
- **NumPy / SciPy** – Audio data processing  

### Frontend
- **HTML, CSS, JavaScript** – User interface and interactivity  
- **MediaRecorder API** – Captures live microphone input  
- **Fetch API** – Sends audio to the Flask server for inference  


## How It Works

1. **Voice Enrollment**
   - Collects and stores embeddings for known speakers.  
2. **Real-Time Identification**
   - User clicks “Start Recording”.  
   - The system records a 15-second clip and sends it to the backend.  
   - ECAPA-TDNN extracts embeddings and matches them with stored speakers.  
   - The result (speaker name + confidence score) is displayed instantly.


