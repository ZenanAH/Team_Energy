# streamlit app structure file

# import libraries
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
import plotly as py

# Page config
st.set_page_config(page_title="Test Streamlit App", page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")

# Use local css file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style/style.css")

# fucntion to load lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Assest inport and assignment to variables
lottie_coding_Pyraminx_Shape_Lottie_Animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_2QAV9ZfHr4.json")
lottie_coding_Data_Science_Animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_xl3sktpc.json")

# Header section
with st.container():
    st.title("Streamlit App")
    st.subheader("This is a Test web app for LeWagon Team Energy")
    st.write("This is a test app for LeWagon Team Energy to test Streamlit")
    st.write("[Le Wagon Home Page](https://www.lewagon.com)")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.write('''
        This is the left column.
        it is 50% of the page but
        im sure we coudl make it bigger
        ''')
    with right_column:
        st_lottie(lottie_coding_Pyraminx_Shape_Lottie_Animation, height=200, key="coding")

with st.container():
    st.write("---")
    st_lottie(lottie_coding_Data_Science_Animation, height=200)

# graphst section -- Plotly
with st.container():
    st.write("---")
    st.write("This is a plotly graph")

# contact form section from form submit
# link for info here --> https://formsubmit.co/
with st.container():
    st.write("---")
    contact_form = '''
    <form action="https://formsubmit.co/jordan.lee.harris@icloud.com" method="POST">
        <input type="hidden" name="captcha" value="false" />
        <input type="text" name="name" required>
        <input type="email" name="email" required>
        <textarea name="Anything else you would like to say?" required></textarea>
        <button type="submit">Send</button>
    </form>
    '''

left_column, right_column = st.columns(2)
with left_column:
    st.markdown(contact_form, unsafe_allow_html=True)
with right_column:
    st.empty()

# footer section with contrabuters and links
with st.container():
    st.write("---")
    st.write("Built badly by Jordan Harris")
    st.write("Lets hope team energy can do better")