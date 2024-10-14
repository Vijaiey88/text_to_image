import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the pre-trained Stable Diffusion model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"  

# Enable GPU support if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    return pipe.to(device)

pipe = load_pipeline()

# Streamlit app title
st.title("Text-to-Image Generator with Stable Diffusion")

# Text input for the prompt
prompt = st.text_input("Enter a text prompt for image generation:", "A futuristic cityscape at sunset, digital art")

if st.button("Generate Image"):
    if prompt:
        with st.spinner('Generating image...'):
            # Generates image based on the text prompt
            image = pipe(prompt).images[0]
            
            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)
            # Save the image for reference
            image.save("generated_image.png")
    else:
        st.error("Please enter a text prompt to generate an image.")
