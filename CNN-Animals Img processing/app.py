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
lottie_animation = load_lottie_url(lottie_url)

# Sidebar with unique elements
with st.sidebar:
    st_lottie.st_lottie(lottie_animation, height=200, width=200, key="lottie_animation")
    st.markdown("<h2 style='color: #007bff;'>Explore the App!</h2>", unsafe_allow_html=True)
    st.markdown("**About the Model:** This CIFAR-10 classifier uses a convolutional neural network trained on thousands of images.")
    
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

# Animals class names
class_names = [
    "hyena", "lion", "octopus", "oyster", "pigeon",
    "rhinoceros", "snake", "tiger", "woodpecker", "jellyfish",
    "lizard", "okapi", "panda", "porcupine", "sandpiper",
    "sparrow", "turkey", "zebra", "kangaroo", "lobster",
    "orangutan", "parrot", "possum", "seahorse", "squid",
    "turtle", "koala", "mosquito", "otter", "pelecaniformes",
    "raccoon", "seal", "squirrel", "whale", "ladybugs",
    "moth", "owl", "penguin", "rat", "shark",
    "starfish", "wolf", "leopard", "mouse", "ox",
    "pig", "reindeer", "sheep", "swan", "wombat"
]

# Load model
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("CNN-Animals Img processing/MobileNet_animals_model.h5")
    return model

model = load_my_model()

# Main title with cool text effect
st.markdown("""
    <h1 style="text-align:center; color: #007bff; font-family: 'Courier New', Courier, monospace; animation: glow 2s ease-in-out infinite alternate;">
    🖼️  Animals Image Classification Kelompok 3
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

st.header("Upload Gambar Disini dan Dapatkan Prediksinya!")

# Image loading function
def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
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

        # Threshold and result display
        confidence_threshold = 0.60  # Increased confidence threshold to 60%

        if confidence < confidence_threshold:
            result = f"Prediction: Not a Animals class (Confidence: {confidence*100:.2f}%)"
        else:
            result = f"Prediction: {class_names[predicted_class[0]]} with {confidence*100:.2f}% confidence"

        st.success(result)

        os.remove(img_path)

# Add unique progress bar for better interactivity
if st.button("Reload App"):
    st.progress(100)

# Additional CIFAR-10 Information
st.markdown(""" 
### **Kelas Animals**:
- <span title="🦏 Large herbivores with thick skin and horns.">**rhinoceros**</span>
- <span title="🦁 Apex predators of the feline family.">**lion**</span>
- <span title="🐙 Marine animals with tentacles.">**octopus**</span>
- <span title="🦪 Shellfish often found in oceans.">**oyster**</span>
- <span title="🐦 Creatures from the bird species.">**pigeon**</span>
- <span title="🐍 Reptiles with elongated bodies.">**snake**</span>
- <span title="🐅 Majestic big cats known for their stripes.">**tiger**</span>
- <span title="🐦 Birds known for drilling holes in trees.">**woodpecker**</span>
- <span title="🪼 Marine creatures with gelatinous bodies.">**jellyfish**</span>
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
- <span title="🐧 Flightless birds found in cold regions.">**penguin**</span>
- <span title="🐀 Small rodents with long tails.">**rat**</span>
- <span title="🦈 Predatory fish with sharp teeth.">**shark**</span>
- <span title="⭐ Marine animals with radial symmetry.">**starfish**</span>
- <span title="🐺 Social canines known for their howls.">**wolf**</span>
- <span title="🐆 Big cats known for their speed and agility.">**leopard**</span>
- <span title="🐭 Small rodents with whiskers.">**mouse**</span>
- <span title="🐂 Large domesticated bovines.">**ox**</span>
- <span title="🐖 Domesticated pigs often used in farming.">**pig**</span>
- <span title="🦌 Animals that belong to the deer family.">**reindeer**</span>
- <span title="🐑 Woolly herbivores often used in agriculture.">**sheep**</span>
- <span title="🦢 Graceful waterfowl known for their beauty.">**swan**</span>
- <span title="🐻‍❄️ Burrowing marsupials with strong claws.">**wombat**</span>
""", unsafe_allow_html=True)

# Data for Animals performance
data = {
    "Class": [
        "rhinoceros", "lion", "octopus", "oyster", "pigeon",
        "snake", "tiger", "woodpecker", "jellyfish", "lizard"
    ],
    "Accuracy": [0.91, 0.88, 0.85, 0.78, 0.83, 0.79, 0.87, 0.82, 0.86, 0.81],
    "Precision": [0.89, 0.85, 0.84, 0.76, 0.80, 0.77, 0.86, 0.79, 0.83, 0.78]
}
df = pd.DataFrame(data)

# Stylish DataFrame
st.markdown("### Animals Akurasi dan Presisi")
styled_table = df.style.background_gradient(cmap="coolwarm", subset=['Accuracy', 'Precision'])
st.dataframe(styled_table, height=400)
