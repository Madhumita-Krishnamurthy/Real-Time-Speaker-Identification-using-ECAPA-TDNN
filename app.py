from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
import os
import torch
import soundfile as sf
from speechbrain.inference.speaker import SpeakerRecognition
from torch.nn.functional import cosine_similarity
import torchaudio
from pydub import AudioSegment  # For audio conversion


app = Flask(__name__)
record_file = "temp.wav"
embed_folder = "enrolled_embeddings"
duration = 15  # seconds
sample_rate = 16000

# Load ECAPA-TDNN model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_recording():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio file received"}), 400

    temp_input = "temp_input_audio"
    audio_file.save(temp_input)

    # Convert to proper WAV PCM (mono, 16kHz) using pydub + ffmpeg
    audio = AudioSegment.from_file(temp_input)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    # ---- Volume Fix ----
    audio = audio.normalize()
    audio.export(record_file, format="wav")
    os.remove(temp_input)

    # Load and resample audio
    signal, fs = sf.read(record_file)
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    if fs != sample_rate:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=sample_rate)(signal)

    # Generate embedding
    test_embedding = model.encode_batch(signal).squeeze()
    test_embedding = test_embedding / test_embedding.norm(p=2)

    # Compare with enrolled embeddings
    max_sim = float("-inf")
    best_match = None
    for file in os.listdir(embed_folder):
        if file.endswith(".pt"):
            name = file.replace(".pt", "")
            enrolled_embedding = torch.load(os.path.join(embed_folder, file))
            enrolled_embedding = enrolled_embedding / enrolled_embedding.norm(p=2)
            sim = cosine_similarity(test_embedding, enrolled_embedding, dim=0).item()
            if sim > max_sim:
                max_sim = sim
                best_match = name

    similarity = round(max_sim * 100, 2)

    if similarity < 30:
        identified_name = "Unknown"
        image_filename = "default.jpg"
        response = {
            "identified_name": identified_name,
            "audio_path": "/audio",
            "image_url": f"/images/{image_filename}"
        }
    else:
        identified_name = best_match
        image_filename = f"{identified_name}.jpg"
        if not os.path.exists(os.path.join("images", image_filename)):
            image_filename = "default.jpg"
        response = {
            "identified_name": identified_name,
            "similarity": similarity,
            "audio_path": "/audio",
            "image_url": f"/images/{image_filename}"
        }

    return jsonify(response)


@app.route("/audio")
def serve_audio():
    return send_file(record_file, mimetype="audio/wav")

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)


if __name__ == "__main__":
    app.run(debug=True)
