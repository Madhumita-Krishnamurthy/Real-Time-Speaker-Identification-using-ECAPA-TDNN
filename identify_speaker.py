import os
import torch
import sounddevice as sd
import soundfile as sf
from speechbrain.inference.speaker import SpeakerRecognition

# Parameters for recording
duration = 15           # seconds to record
sample_rate = 16000     # model expects 16kHz audio
record_file = "temp.wav"
embed_folder = "enrolled_embeddings"

# Load pretrained model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Record audio from microphone
print(f"🎙️ Please speak now... Recording for {duration} seconds.")
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
sd.wait()
sf.write(record_file, recording, sample_rate)
print("✅ Recording complete.")

# Load recorded audio
signal, fs = sf.read(record_file)
signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

# Resample if needed
if fs != sample_rate:
    import torchaudio
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=sample_rate)
    signal = resampler(signal)

# Extract embedding for recorded audio
test_embedding = model.encode_batch(signal).squeeze()
# Normalize test embedding as well (important!)
test_embedding = test_embedding / test_embedding.norm(p=2)

# Compare with centroid embeddings
max_sim = float("-inf")
identified_name = "Unknown"

for file in os.listdir(embed_folder):
    if file.endswith(".pt"):
        name = file.replace(".pt", "")
        enrolled_embedding = torch.load(os.path.join(embed_folder, file))
        # Ensure embedding is normalized (if not, normalize here)
        enrolled_embedding = enrolled_embedding / enrolled_embedding.norm(p=2)
        sim = torch.nn.functional.cosine_similarity(test_embedding, enrolled_embedding, dim=0)
        if sim > max_sim:
            max_sim = sim
            identified_name = name

print(f"🔍 Identified speaker: **{identified_name}** (Similarity: {max_sim:.4f})")
