import streamlit as st
import time
from langchain_pipe import return_agent
import streamlit_antd_components as sac
import base64
from pathlib import Path
from openai import OpenAI as OP
import pyaudio
import wave
import google.generativeai as genai 
import google.ai.generativelanguage as glm 
from dotenv import load_dotenv
from PIL import Image
import os 
import io 
import pandas as pd



def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr=imgByteArr.getvalue()
    return imgByteArr

API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

load_dotenv()

# ------------------------------------------- Record Voice Notes ----------------------------------------------------------

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


# ------------------------------------------- Streamed response emulator ----------------------------------------------------------

def response_generator(agent, prompt):
    response_dict = agent(prompt)
    response = response_dict["output"]
    print(response)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

api_key = st.secrets["OPEN_API_KEY"]

agent = return_agent(api_key)

vopenai = OP(
    api_key=st.secrets["OPEN_API_KEY"]
)

# Define your page functions
def home_page():
    
    with open('style.css') as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    # Display HTML content
    st.markdown("""
    <div class="area" >
        <ul class="circles">
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
            <li></li>
        </ul>
                
    </div >
    """, unsafe_allow_html=True)
    
    st.session_state['b64_image'] =""
    with open("./pic.png", "rb") as img_file:
        img_back = base64.b64encode(img_file.read()).decode("utf-8")
        # st.image(f'data:image/png;base64,{img_back}', use_column_width=False)
        st.markdown(f"""<img class="back_img"  src="data:image/png;base64,{img_back}" alt="Frozen Image">""",unsafe_allow_html=True)
    st.markdown("""<h1 class="Title">Welcome To CogniSmile</h1>""",unsafe_allow_html=True)

def generate_speech(input_text):
    speech_file_path = Path("speech.mp3")
    response = vopenai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=input_text
    )
    response.stream_to_file(speech_file_path)

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true" style="display: none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
            
        )
        
def generateTextFromVoice(path):
    audio_file= open(path, "rb")
    transcription = vopenai.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    return transcription.text


def Assistant():
    st.title("DAIT Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
    
    
    
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = "".join(response_generator(agent, prompt))
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        generate_speech(response)
        autoplay_audio("speech.mp3")
    if st.button("Record"):
            record_audio()
            VoiceMessage = generateTextFromVoice("./output.wav")
            with st.chat_message("user"):
                st.markdown(VoiceMessage)
                
            with st.chat_message("assistant"):
                response = "".join(response_generator(agent, VoiceMessage))
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        
            generate_speech(response)
            autoplay_audio("speech.mp3")
            
        
def generate_story():
    return 0 

def Food_Helper():
    def get_gemini_response(input_prompt, image):
        model=genai.GenerativeModel('gemini-pro-vision')
        response=model.generate_content([input_prompt, image[0]])
        return response.text

    def input_image_setup(uploaded_file):
        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Read the file into bytes
            bytes_data = uploaded_file.getvalue()

            image_parts = [
                {
                    "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No file uploaded")
        
    ##initialize our streamlit app


    st.header("Food Helper")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)


    submit=st.button("Tell me about the  glycemic index")

    input_prompt="""
    You are an expert in  meal prep AI assistant for diabetics where you need to see the food items from the image
                and calculate the total glycemic index in mg/dl, also provide the details of every food items with  glycemic index  intake
                is below format

                1. Item 1 - no of  glycemic index  in mg/dl 
                2. Item 2 - no of  glycemic index  in mg/dl
                ---- 
                ----
    Finally you can also mention whether the food is healthy, balanced or not healthy and what all additional food items can be added in the diet which are healthy.

    """

    ## If submit button is clicked

    if submit:
        image_data=input_image_setup(uploaded_file)
        response=get_gemini_response(input_prompt,image_data)
        st.write(response)
    

def Statistics():
    # Existing sample data
    data = """FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 07:12 PM,0,457,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 07:27 PM,0,459,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 07:42 PM,0,455,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 07:57 PM,0,437,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 08:13 PM,0,414,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 08:28 PM,0,426,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 08:43 PM,0,433,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 08:58 PM,0,411,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 09:13 PM,0,383,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 09:28 PM,0,357,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 09:43 PM,0,322,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 09:58 PM,0,280,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 10:13 PM,0,231,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 10:28 PM,0,182,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 10:43 PM,0,146,,,,,,,,,,,,,,"""

    # Additional data to be added
    additional_data = """FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 10:58 PM,0,100,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 11:13 PM,0,90,,,,,,,,,,,,,,
    FreeStyle LibreLink,10D06121-1654-4FE8-8850-08C1E630EF7C,07-29-2022 11:28 PM,0,85,,,,,,,,,,,,,,"""

    # Concatenate additional data to the existing data
    data += additional_data

    # Function to process data
    def process_data(data):
        lines = data.splitlines()
        data_list = [line.split(",") for line in lines]
        dates = [row[2].strip() for row in data_list]
        glucose_values = [int(row[4]) for row in data_list]
        max_glucose = max(glucose_values)
        min_glucose = min(glucose_values)
        df = pd.DataFrame({"Date": dates, "Glucose Value": glucose_values})
        # Optional: Convert "Date" to datetime format
        df["Date"] = pd.to_datetime(df["Date"], format='%m-%d-%Y %I:%M %p')
        return df,max_glucose,min_glucose

    # Process the data
    df, max_glucose, min_glucose= process_data(data)
    last_glucose_value = df.iloc[-1]["Glucose Value"]
    percentage1=(100-last_glucose_value)/100
    col1, col2, col3 = st.columns(3)
    col1.metric("highest glucose level", str(max_glucose) +" mg/dL", "max%")
    col2.metric("lower glucose level", str(min_glucose) +" mg/dL", "-min%")
    col3.metric("Current glucose level", str(last_glucose_value)+" mg/dL", str(percentage1)+"%")
    # Title and header
    st.title("Blood Glucose Monitoring Chart")
    st.header("FreeStyle LibreLink Data")



    # Display the line chart
    st.line_chart(df, x="Date", y="Glucose Value")

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Home'

with st.sidebar:
    selected_tab = sac.menu([
        sac.MenuItem('Home', icon='house-fill'),
        sac.MenuItem('Assistant', icon='chat-text-fill'),
        sac.MenuItem('Food-Helper', icon='bi bi-award-fill'),
        sac.MenuItem('Statistics', icon='easel2-fill')
    ], color='cyan', size='lg', open_all=True)

if selected_tab != st.session_state.current_tab:
    st.session_state.current_tab = selected_tab

if st.session_state.current_tab == 'Home':
    home_page()
elif st.session_state.current_tab == 'Assistant':
    Assistant()  
elif st.session_state.current_tab == 'Food-Helper':
    Food_Helper()
elif st.session_state.current_tab == 'Statistics':
    Statistics()        
