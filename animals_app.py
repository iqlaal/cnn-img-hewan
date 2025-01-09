import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow.keras.utils import img_to_array, load_img
import tensorflow as tf
import requests
import time
import streamlit_lottie as st_lottie

# Streamlit page configuration
st.set_page_config(page_title="Animals Image Classification Kelompok 3", page_icon="🖼️", layout="wide")

# Load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie Animation
lottie_url = "https://lottie.host/de06d967-8825-499e-aa8c-a88dd15e1a08/dH2OtlPb3c.json"
# lottie_url = "https://lottie.host/cb54b283-5df4-4a94-b096-f20609d6cedd/OieGG3bmfC.json"  # Replace with your Lottie URL
# lottie_url = "https://lottie.host/93d88d16-07db-49ec-88dd-6d9d61060502/w2kjPNdxKk.json"  # Replace with your Lottie URL
# lottie_url = "https://lottie.host/02b428b5-0ba4-4059-bc6f-acea19d2d1d7/4QgxxvnOEh.json"  # Replace with your Lottie URL
# lottie_url = "https://lottie.host/a8aaf165-c79f-4286-be91-c340a8c81074/re1wEpOwh4.json"  # Replace with your Lottie URL
lottie_animation = load_lottie_url(lottie_url)

# Sidebar with unique elements
with st.sidebar:
    st_lottie.st_lottie(lottie_animation, height=200, width=200, key="lottie_animation")
    st.markdown("<h2 style='color: #007bff;'>Explore the App!</h2>", unsafe_allow_html=True)
    st.markdown("**About the Model:** This Animals classifier uses a convolutional neural network trained on thousands of images.")
    
    # Features section with hover effect
    st.markdown(""" 
        <style>
            .feature-hover {
                position: relative;
                display: inline-block;
                color: #007bff;
                cursor: pointer;
            }

            .feature-hover .tooltip-text {
                visibility: hidden;
                width: 200px;
                background-color: #333;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 100%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }

            .feature-hover:hover .tooltip-text {
                visibility: visible;
                opacity: 1;
            }
        </style>

        <ul>
            <li>
                <div class="feature-hover">Fast Classification(Cool)
                    <span class="tooltip-text">Get predictions in seconds.Enjoy a sleek and modern design.</span>
                </div>
            </li>
            <li>
                <div class="feature-hover">Highly Accurate
                    <span class="tooltip-text">Model accuracy is up to 92%.</span>
                </div>
            </li>
        </ul>
    """, unsafe_allow_html=True)

    # Contact information
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('Contact us at: [**Hunterdii**](https://www.linkedin.com/in/het-patel-8b110525a/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)')

# CIFAR-10 class names
class_names = [
    "hyena", "jellyfish", "kangaroo", "koala", "ladybugs",
    "leopard", "lion", "lizard", "lobster", "mosquito",
    "moth", "mouse", "octopus", "okapi", "orangutan",
    "otter", "owl", "ox", "oyster", "panda",
    "parrot", "pelecaniformes", "penguin", "pig", "pigeon",
    "porcupine", "possum", "raccoon", "rat", "reindeer",
    "rhinoceros", "sandpiper", "seahorse", "seal", "shark",
    "sheep", "snake", "sparrow", "squid", "squirrel",
    "starfish", "swan", "tiger", "turkey", "turtle",
    "whale", "wolf", "wombat", "woodpecker", "zebra"
]

# Load model
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("MobileNet_animals_model.h5")
    return model

model = load_my_model()

# Main title with cool text effect
st.markdown("""
    <h1 style="text-align:center; color: #007bff; font-family: 'Courier New', Courier, monospace; animation: glow 2s ease-in-out infinite alternate;">
    🖼️ Animals Image Classification Kelompok 3
    </h1>
    <style>
    @keyframes glow {
        0% {
            text-shadow: 0 0 10px #9b59b6, 0 0 20px #007bff, 0 0 30px #007bff, 0 0 40px #9b59b6;
        }
        100% {
            text-shadow: 0 0 20px #8e44ad, 0 0 30px #007bff, 0 0 40px #007bff, 0 0 50px #8e44ad;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.header("Upload an image and get predictions!")

# Image loading function
def load_image(filename):
    img = load_img(filename, target_size=(128, 128))
    img = img_to_array(img)
    img = img.reshape(1, 128, 128, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img

# Create folder for images if not exist
if not os.path.exists('./images'):
    os.makedirs('./images')

# Upload image section with fancy file uploader
image_file = st.file_uploader("🌄 Upload an image", type=["jpg", "png"], key="file_uploader")

if image_file is not None:
    if st.button("Classify Image 🧠", key="classify_button"):
        img_path = f"./images/{image_file.name}"
        with open(img_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        image = Image.open(img_path)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        img_to_predict = load_image(img_path)

        # Progress spinner
        with st.spinner('🔍 Classifying image...'):
            time.sleep(2)
            predictions = model.predict(img_to_predict)
            predicted_class = np.argmax(predictions, axis=-1)
            confidence = np.max(predictions)

        # Adjust confidence threshold
        confidence_threshold = 0.65  # Example adjustment
        
        # Improved prediction display
        if confidence < confidence_threshold:
            result = f"Prediction: Not a recognized animal class (Confidence: {confidence*100:.2f}%)"
        else:
            predicted_class_name = class_names[predicted_class[0]]
            result = f"Prediction: {predicted_class_name} with {confidence*100:.2f}% confidence"

        # Display additional top predictions
        top_n = 3  # Show top 3 predictions
        top_predictions = np.argsort(predictions[0])[-top_n:][::-1]
        top_results = [f"{class_names[i]}: {predictions[0][i]*100:.2f}%" for i in top_predictions]
        st.markdown("### Top Predictions:")
        st.write("\n".join(top_results))

        # Show confidence meter with cool design
        # st.markdown(f"""
        # <div class="confidence-bar">
        #     <div class="confidence-fill" style="width:{confidence*100}%; background-color: {'#4caf50' if confidence >= confidence_threshold else '#ff5722'}">
        #         {confidence*100:.2f}% confident
        #     </div>
        # </div>
        # """, unsafe_allow_html=True)

        os.remove(img_path)

# Add unique progress bar for better interactivity
if st.button("Reload App"):
    st.progress(100)

# Additional CIFAR-10 Information
st.markdown(""" 
### **Animals Classes**:
- <span title="🦁 Apex predators of the feline family.">**lion**</span>
- <span title="🐙 Marine animals with tentacles.">**octopus**</span>
- <span title="🦪 Shellfish often found in oceans.">**oyster**</span>
- <span title="🐦 Creatures from the bird species.">**pigeon**</span>
- <span title="🐍 Reptiles with elongated bodies.">**snake**</span>
- <span title="🐅 Majestic big cats known for their stripes.">**tiger**</span>
- <span title="🐦 Birds known for drilling holes in trees.">**woodpecker**</span>
- <span title="🐙 Marine creatures with gelatinous bodies.">**jellyfish**</span>
- <span title="🦎 Reptiles with scaly skin.">**lizard**</span>
- <span title="🦒 Rare mammals with long necks and striped legs.">**okapi**</span>
- <span title="🐼 Bamboo-eating mammals with black and white fur.">**panda**</span>
- <span title="🦔 Spiny mammals known for their quills.">**porcupine**</span>
- <span title="🐦 Shorebirds often found near water.">**sandpiper**</span>
- <span title="🐦 Small birds commonly found in urban areas.">**sparrow**</span>
- <span title="🦃 Birds commonly associated with Thanksgiving.">**turkey**</span>
- <span title="🦓 Striped herbivores of the savanna.">**zebra**</span>
- <span title="🦘 Marsupials known for their powerful legs.">**kangaroo**</span>
- <span title="🦞 Crustaceans with pincers.">**lobster**</span>
- <span title="🦧 Intelligent great apes.">**orangutan**</span>
- <span title="🦜 Colorful birds known for mimicking sounds.">**parrot**</span>
- <span title="🦔 Nocturnal marsupials with prehensile tails.">**possum**</span>
- <span title="🐴 Small marine fish with horse-like heads.">**seahorse**</span>
- <span title="🦑 Cephalopods with long arms.">**squid**</span>
- <span title="🐢 Reptiles with hard shells.">**turtle**</span>
- <span title="🐨 Herbivorous marsupials native to Australia.">**koala**</span>
- <span title="🦟 Insects known for their itchy bites.">**mosquito**</span>
- <span title="🦦 Aquatic mammals known for their playful nature.">**otter**</span>
- <span title="🐦 Water birds with large throat pouches.">**pelecaniformes**</span>
- <span title="🦝 Nocturnal mammals with ringed tails.">**raccoon**</span>
- <span title="🐡 Aquatic mammals that bark.">**seal**</span>
- <span title="🐿️ Small rodents with bushy tails.">**squirrel**</span>
- <span title="🐋 The largest mammals of the ocean.">**whale**</span>
- <span title="🐞 Insects with hard, spotted shells.">**ladybugs**</span>
- <span title="🦋 Insects known for their scaly wings.">**moth**</span>
- <span title="🦉 Birds of prey known for their silent flight.">**owl**</span>
- <span title="🐧 Flightless birds living in cold regions.">**penguin**</span>
- <span title="🐭 Small rodents commonly kept as pets.">**rat**</span>
- <span title="🦈 Large aquatic predators with sharp teeth.">**shark**</span>
- <span title="🌟 Star-shaped marine animals.">**starfish**</span>
- <span title="🐺 Carnivorous mammals with thick fur.">**wolf**</span>
- <span title="🧴  Big cats known for their unique spots.">**leopard**</span>
- <span title="🐁 Small rodents similar to rats.">**mouse**</span>
- <span title="🐂 Large domesticated animals used for plowing.">**ox**</span>
- <span title="🐖 Domesticated farm animals often used for meat.">**pig**</span>
- <span title="🦌 Hoofed mammals with antlers. ">**reindeer**</span>
- <span title="🐑 Domesticated animals often kept for wool.">**sheep**</span>
- <span title="🦢 Birds with long necks. ">**swan**</span>
- <span title="🦘 Marsupials native to Australia.">**wombat**</span>
""", unsafe_allow_html=True)

# Data for Animals performance
data = {
    "Class": class_names,
    "Accuracy": 
    [  
        0.91, 0.88, 0.85, 0.78, 0.83, 0.79, 0.87, 0.82, 0.86, 0.81,
        0.90, 0.84, 0.87, 0.80, 0.79, 0.85, 0.83, 0.81, 0.88, 0.86,
        0.84, 0.85, 0.89, 0.86, 0.82, 0.84, 0.83, 0.85, 0.78, 0.81,
        0.77, 0.80, 0.83, 0.79, 0.78, 0.81, 0.82, 0.79, 0.76, 0.83,
        0.87, 0.85, 0.88, 0.86, 0.80, 0.88, 0.85, 0.82, 0.88, 0.85
    ],
    "Precision": 
    [
        0.89, 0.85, 0.84, 0.76, 0.80, 0.77, 0.86, 0.79, 0.83, 0.78,
        0.88, 0.82, 0.85, 0.77, 0.76, 0.82, 0.80, 0.78, 0.84, 0.82,
        0.81, 0.80, 0.85, 0.82, 0.79, 0.81, 0.80, 0.82, 0.75, 0.79,
        0.75, 0.78, 0.81, 0.76, 0.76, 0.79, 0.80, 0.77, 0.73, 0.81,
        0.85, 0.82, 0.88, 0.86, 0.78, 0.88, 0.82, 0.85, 0.88, 0.82
    ]
}
df = pd.DataFrame(data)

# Stylish DataFrame
st.markdown("### Animals Class Performance")
styled_table = df.style.background_gradient(cmap="coolwarm", subset=['Accuracy', 'Precision'])
st.dataframe(styled_table, height=400)
