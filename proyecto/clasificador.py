import time
import numpy as np
import librosa
import pickle
import sounddevice as sd
from tensorflow.keras.models import load_model
from queue import Queue

# Configuración
SAMPLE_RATE = 22050
DURATION = 2  # segundos
SAMPLES = SAMPLE_RATE * DURATION

# Cargar modelo y label encoder
model = load_model('best_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

class RealTimeClassifier:
    def __init__(self):
        self.audio_queue = Queue()
        
    def preprocess(self, audio):
        # Asegurar longitud correcta
        if len(audio) > SAMPLES:
            audio = audio[:SAMPLES]
        else:
            padding = SAMPLES - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Extraer características
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return np.expand_dims(log_mel[..., np.newaxis], axis=0)
    
    def predict(self, audio):
        processed = self.preprocess(audio)
        preds = model.predict(processed, verbose=0)
        return label_encoder.inverse_transform([np.argmax(preds)])[0], np.max(preds)
    
    def callback(self, indata, frames, time, status):
        """Se llama para cada bloque de audio"""
        self.audio_queue.put(indata.copy())
        
    def start(self):
        print("Iniciando clasificación en tiempo real... (Presiona Ctrl+C para detener)")
        try:
            with sd.InputStream(callback=self.callback,
                              channels=1,
                              samplerate=SAMPLE_RATE,
                              blocksize=SAMPLE_RATE//4):  # Bloques de 0.25s
                
                buffer = np.zeros((0, 1))
                while True:
                    # Obtener audio del micrófono
                    while not self.audio_queue.empty():
                        buffer = np.concatenate((buffer, self.audio_queue.get()))
                    
                    # Cuando tenemos suficiente audio, procesar
                    if len(buffer) >= SAMPLES:
                        audio_chunk = buffer[:SAMPLES]
                        buffer = buffer[SAMPLES//2:]  # Solapamiento del 50%
                        
                        # Predecir y mostrar resultados
                        label, confidence = self.predict(audio_chunk.flatten())
                        print(f"Predicción: {label} (Confianza: {confidence:.2%})\n", end='\r')
                    
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nClasificación detenida")

# Iniciar
if __name__ == "__main__":
    classifier = RealTimeClassifier()
    classifier.start()