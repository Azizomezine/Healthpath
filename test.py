import streamlit as st
import pyaudio
import wave
import numpy as np

def record_audio(seconds=5, rate=44100, channels=1):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    for i in range(int(rate / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert frames to binary data
    wave_output = wave.open("output.wav", 'wb')
    wave_output.setnchannels(channels)
    wave_output.setsampwidth(audio.get_sample_size(FORMAT))
    wave_output.setframerate(rate)
    wave_output.writeframes(b''.join(frames))
    wave_output.close()

    return b''.join(frames)

def audiorec_demo_app():
    st.title('Streamlit Audio Recorder')

    # Record audio
    st.write("Click the button below to start recording:")
    if st.button("Record"):
        st.write("Recording...")
        audio_data = record_audio()
        st.write("Recording stopped.")
        st.audio(audio_data, format='audio/wav')

if __name__ == '__main__':
    audiorec_demo_app()
