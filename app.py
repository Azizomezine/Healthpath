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
from datetime import datetime, timedelta  # Import datetime and timedelta explicitly


def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr=imgByteArr.getvalue()
    return imgByteArr

API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

load_dotenv()
st.markdown(
        """
        <style>
            {}
        </style>
        """.format(open("style.css").read()),
        unsafe_allow_html=True
    )
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
    # Display HTML content
    # st.markdown("""
    # <div class="area" >
    #     <ul class="circles">
    #         <li></li>
    #         <li></li>
    #         <li></li>
    #         <li></li>
    #         <li></li>
    #         <li></li>
    #         <li></li>
    #         <li></li>
    #         <li></li>
    #         <li></li>
    #     </ul>
                
    # </div >
    # """, unsafe_allow_html=True)
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
    with open("./Back.png", "rb") as img_file:
        img_back = base64.b64encode(img_file.read()).decode("utf-8")
        # st.image(f'data:image/png;base64,{img_back}', use_column_width=False)
        st.markdown(f"""<img class="back_img"  src="data:image/png;base64,{img_back}" alt="Frozen Image">""",unsafe_allow_html=True)
    st.markdown("""<h1 class="Title">Welcome To HealthPath</h1>""",unsafe_allow_html=True)

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
    st.title("HealthPath Assistant")

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
    if st.button("Talk to me"):
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
def get_gemini_response(input_prompt, image):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input_prompt, image[0]])
    return response.text
def get_gemini_pro(input_user):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input_user)
    return response.text
def Food_Helper():
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


    st.header("Determine the right dose of insulin")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""
    ICR=st.text_input("insulin-to-carb ratio (ICR) (grams/unit)")   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)


    submit=st.button("Discover insights")

    input_prompt="""
    You are an expert in meal prep AI assistant for diabetics. 

    **Input:**

    * Image of the meal

    **Output:**

    1. Food Items: List all the food items identified in the image.
    2. Glycemic Index (GI): Provide the glycemic index (GI) in mg/dl for each food item.
    3. Total Carbohydrates (CHO): Calculate the total grams of carbohydrate (CHO) for the entire meal.
    4. Recommended Insulin Dose: 
        * Analyze the image to estimate the total CHO content in the meal. 
        * Based on the user's pre-configured insulin-to-carb ratio (ICR), calculate the recommended insulin dose using the formula: Total CHO (grams) / {ICR} (grams/unit) = Recommended Insulin Dose (units).

    **Example Output:**

    1. Food Items:
        * Apple - 30g
        * Bread - 50g
        * Cheese - 10g

    2. Glycemic Index (GI):
        * Apple - 40 mg/dl
        * Bread - 70 mg/dl
        * Cheese - 0 mg/dl (Cheese has minimal impact on blood sugar)

    3. Total Carbohydrates (CHO):  80 grams (calculated by adding the CHO content of each food item)

    4. Recommended Insulin Dose:  
        * Assuming user's ICR is 1 unit per 10 grams of CHO.
        * Recommended dose = 80 grams / 10 grams/unit = 8 units

    **Additional Information:**

    * Indicate if the meal is balanced or not, considering factors like protein, fat, and fiber content (if possible to estimate from the image).
    * Suggest additional healthy food items that could be added to create a more balanced meal.

    **Notes:**

    * The accuracy of the CHO content estimation and GI values may vary depending on the image quality and ingredient identification.
    * This is for informational purposes only and does not replace professional medical advice.
    """

    ## If submit button is clicked

    if submit:
        image_data=input_image_setup(uploaded_file)
        response=get_gemini_response(input_prompt,image_data)
        st.write(response)
    
    # Load data from CSV file
file_path = "data/MohamedAIT ALI_glucose_3-7-2024.csv"
df = pd.read_csv(file_path)

# Rename column with spaces
df.rename(columns={"Horodatage de l'appareil": "Timestamp"}, inplace=True)

# Convert 'Timestamp' column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')

# Group by timestamp and aggregate
df = df.groupby('Timestamp').first().reset_index()

# Reindex with continuous range of timestamps
min_timestamp = df['Timestamp'].min()
max_timestamp = df['Timestamp'].max()
all_timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq='T')  # Minute frequency
df = df.set_index('Timestamp').reindex(all_timestamps).reset_index()

# Interpolate missing values
df['Historique de la glycémie mg/dL'] = df['Historique de la glycémie mg/dL'].interpolate(method='linear')

# Filter data for the last 200 days
last_200_days = datetime.now() - timedelta(days=30)
df_last_200_days = df[df['index'] >= last_200_days]

# Function to process data
def process_data(df):
    max_glucose = df["Historique de la glycémie mg/dL"].max()
    min_glucose = df["Historique de la glycémie mg/dL"].min()
    last_glucose_value = df.iloc[-1]["Historique de la glycémie mg/dL"]
    percentage1 = (100 - last_glucose_value) / 100
    return df, max_glucose, min_glucose, last_glucose_value, percentage1

    # Process the data
df_last_200_days, max_glucose, min_glucose, last_glucose_value, percentage1 = process_data(df_last_200_days)
def Statistics():

    st.title("Blood Glucose Monitoring Chart")
    col1, col2, col3 = st.columns(3)
    col1.metric("highest glucose level", str(max_glucose) +" mg/dL", "max%")
    col2.metric("lower glucose level", str(min_glucose) +" mg/dL", "-min%")
    col3.metric("Current glucose level", str(last_glucose_value)+" mg/dL", str(percentage1)+"%")
    # Title and header
    st.header("FreeStyle LibreLink Data")
    # Display the line chart
    st.line_chart(df_last_200_days, x="index", y="Historique de la glycémie mg/dL")
    st.header("Physical Activity Recommendation")
    submit=st.button("Discover insights")

    input_prompt="""
    You are developing an AI assistant for individuals managing diabetes.

    **Input:**

     current glucose level in mg/dL ({last_glucose_value}).

    **Output:**

    1. Analyze Glucose Level:
        * Determine if the user's current glucose level ({last_glucose_value} mg/dL) is within a safe range for exercise.
        * If the glucose level is too low (<70 mg/dL) or too high (>250 mg/dL), advise against engaging in strenuous exercise.
        * If the glucose level is within the safe range, provide encouragement to proceed with exercise.

    2. Alternative Exercise Suggestions:
        * If the glucose level is too low or too high for strenuous exercise, suggest alternative activities that are safe and beneficial.
        * For low glucose levels, recommend activities like walking, light stretching, or yoga to avoid further lowering blood sugar.
        * For high glucose levels, recommend activities like brisk walking, cycling, or swimming to help lower blood sugar levels.
        * Provide step-by-step instructions or tips for each recommended activity to ensure safety and effectiveness.

    3. Additional Recommendations:
        * Offer general tips for exercising with diabetes, such as staying hydrated, checking glucose levels before and after exercise, and carrying snacks in case of hypoglycemia.
        * Emphasize the importance of consulting with a healthcare professional before starting any new exercise regimen, especially for individuals with diabetes.

    **Example Output:**

    1. Analyze Glucose Level:
        * Glucose level: {last_glucose_value} mg/dL
        * Within safe range for exercise. Proceed with caution and monitor blood sugar levels.

    2. Alternative Exercise Suggestions:
        * Given the current glucose level, it's safe to engage in activities like brisk walking or light jogging.
        * Step-by-step instructions for brisk walking:
            - Wear comfortable shoes and clothing.
            - Start with a warm-up by walking at a moderate pace for 5-10 minutes.
            - Increase your pace to a brisk walk, maintaining a steady speed.
            - Aim for at least 30 minutes of brisk walking.
            - Cool down by walking at a slower pace for 5-10 minutes.
        * Remember to carry water and glucose tablets in case of low blood sugar.

    3. Additional Recommendations:
        * Check glucose levels before and after exercise to monitor the impact on blood sugar.
        * Consider consulting with a certified diabetes educator or fitness trainer for personalized exercise recommendations.
    """

    ## If submit button is clicked

    if submit:
        response=get_gemini_pro(input_prompt)
        st.write(response)
        

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
