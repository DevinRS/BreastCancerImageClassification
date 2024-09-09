import streamlit as st
import glob
from streamlit_carousel import carousel
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from models.cellcount import canny_count, hough_circle_count, watershed_detection
import numpy as np

# region: -- SESSION STATE --
with st.spinner("Loading Models..."):
    if "model_xception" not in st.session_state:
        st.session_state.model_xception = load_model("models/weights_1/best_model_xception_finetuned.keras")

    if "model_resnet50v2" not in st.session_state:
        st.session_state.model_resnet50v2 = load_model("models/weights_1/best_model_resnet50v2_finetuned.keras")

    if "model_inceptionresnetv2" not in st.session_state:
        st.session_state.model_inceptionresnetv2 = load_model("models/weights_1/best_model_inceptionresnetv2_finetuned.keras")

    if "model_densenet201" not in st.session_state:
        st.session_state.model_densenet201 = load_model("models/weights_1/best_model_densenet201_finetuned.keras")

    if "model_efficientnetb4" not in st.session_state:
        st.session_state.model_efficientnetb4 = load_model("models/weights_1/best_model_efficientnetb4_finetuned.keras")

    if "model_efficientnetv2s" not in st.session_state:
        st.session_state.model_efficientnetv2s = load_model("models/weights_1/best_model_efficientnetv2s_finetuned.keras")

    if "results_ER" not in st.session_state:
        st.session_state.results_ER = []

    if "results_PR" not in st.session_state:
        st.session_state.results_PR = []

    if "results_HER2" not in st.session_state:
        st.session_state.results_HER2 = []

    if "results_KI67" not in st.session_state:
        st.session_state.results_KI67 = []

    if "count_ER" not in st.session_state:
        st.session_state.count_ER = []

    if "count_PR" not in st.session_state:
        st.session_state.count_PR = []

    if "count_HER2" not in st.session_state:
        st.session_state.count_HER2 = []

    if "count_KI67" not in st.session_state:
        st.session_state.count_KI67 = []

    if "prev_patient_id" not in st.session_state:
        st.session_state.prev_patient_id = 0
# endregion: -- SESSION STATE --

# region: -- UTILITIES --
# Function to get the patient image given the patient ID
def get_patient_image(patient_id, biomarker=None):
    images = []
    if biomarker is None:
        biomarker = ["ER", "PR", "HER2", "KI67"]
    else:
        biomarker = [biomarker]
    for b in biomarker:
        # Search for files that start with the patient_id in the respective biomarker folder
        search_pattern = f"assets/{b}/{patient_id}.*"
        matching_files = glob.glob(search_pattern)
        images.extend(matching_files)
    return images

# Function to create the carousel items
def create_carousel(title, text, img):
    carousel_items = []
    # create dictionaries from the lists input
    for i in range(len(title)):
        carousel_items.append({"title": title[i], "text": text[i], "img": img[i]})
    return carousel_items

# Function to run inference on the images 
# MAKE THIS INTO API CALL
def run_inference(weights, images):
    model_xception = st.session_state.model_xception

    model_resnet50v2 = st.session_state.model_resnet50v2

    model_inceptionresnetv2 = st.session_state.model_inceptionresnetv2

    model_densenet201 = st.session_state.model_densenet201

    model_efficientnetb4 = st.session_state.model_efficientnetb4

    model_efficientnetv2s = st.session_state.model_efficientnetv2s

    # run inference on the images
    results = []
    for img_path in images:
        # Xception model
        input_size = (299, 299)
        img = image.load_img(img_path, target_size=input_size)  # Adjust target size as needed
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        xception_pred = model_xception.predict(img_array)

        # resnet50v2 model
        input_size = (224, 224)
        img = image.load_img(img_path, target_size=input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        resnet50v2_pred = model_resnet50v2.predict(img_array)

        # inceptionresnetv2 model
        input_size = (299, 299)
        img = image.load_img(img_path, target_size=input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        inceptionresnetv2_pred = model_inceptionresnetv2.predict(img_array)

        # densenet201 model
        input_size = (224, 224)
        img = image.load_img(img_path, target_size=input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        densenet201_pred = model_densenet201.predict(img_array)

        # efficientnetb4 model
        input_size = (384, 380)
        img = image.load_img(img_path, target_size=input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        efficientnetb4_pred = model_efficientnetb4.predict(img_array)

        # efficientnetv2s model
        input_size = (384, 384)
        img = image.load_img(img_path, target_size=input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        efficientnetv2s_pred = model_efficientnetv2s.predict(img_array)

        combined_pred = (xception_pred + resnet50v2_pred + inceptionresnetv2_pred + densenet201_pred + efficientnetb4_pred + efficientnetv2s_pred) / 6

        result = {
            'image': img_path,
            'xception': xception_pred,
            'resnet50v2': resnet50v2_pred,
            'inceptionresnetv2': inceptionresnetv2_pred,
            'densenet201': densenet201_pred,
            'efficientnetb4': efficientnetb4_pred,
            'efficientnetv2s': efficientnetv2s_pred,
            'combined': combined_pred
        }

        results.append(result)

    return results
        

# endregion: -- UTILITIES --

# region: -- BODY --
st.title("Breast Cancer Image Prediction")

# region: -- IMAGE PREVIEW --
patient_id = st.text_input("Patient ID", help="Enter the patient ID", placeholder="Example: 12", value=0, key="patient_id")
if st.session_state.prev_patient_id != patient_id:
    st.session_state.prev_patient_id = patient_id
    st.session_state.results_ER = []
    st.session_state.results_PR = []
    st.session_state.results_HER2 = []
    st.session_state.results_KI67 = []
    st.session_state.count_ER = []
    st.session_state.count_PR = []
    st.session_state.count_HER2 = []
    st.session_state.count_KI67 = []
patient_image = get_patient_image(patient_id)
preview_image = st.checkbox("Preview Image", value=True, help="Check to preview the image", key="preview_image")
if preview_image and patient_image:
    col1, col2, col3, col4, col5 = st.columns(5)
    for i in range(len(patient_image)):
        if i == 0:
            col1.image(patient_image[i], use_column_width=True)
        elif i == 1:
            col2.image(patient_image[i], use_column_width=True)
        elif i == 2:
            col3.image(patient_image[i], use_column_width=True)
        elif i == 3:
            col4.image(patient_image[i], use_column_width=True)
        elif i == 4:
            col5.image(patient_image[i], use_column_width=True)
st.write("---")
# endregion: -- IMAGE PREVIEW --

# region: -- HOMOGEN BIOMARKER CLASSIFICATION --
st.subheader("Homogen Biomarker Classification")
biomarker = st.radio("Select the biomarker to classify the patient image.", ["ER", "PR", "HER2", "KI67"], horizontal=True, key="biomarker")
ER_image = get_patient_image(patient_id, biomarker="ER") # MAKE THIS A SESSION STATE
PR_image = get_patient_image(patient_id, biomarker="PR") # MAKE THIS A SESSION STATE
HER2_image = get_patient_image(patient_id, biomarker="HER2") # MAKE THIS A SESSION STATE
KI67_image = get_patient_image(patient_id, biomarker="KI67") # MAKE THIS A SESSION STATE
# {biomarker}_image contain the file path of the images, extract the file name as the text
ER_text = [img.split("/")[-1] for img in ER_image] # MAKE THIS A SESSION STATE
PR_text = [img.split("/")[-1] for img in PR_image] # MAKE THIS A SESSION STATE
HER2_text = [img.split("/")[-1] for img in HER2_image] # MAKE THIS A SESSION STATE
KI67_text = [img.split("/")[-1] for img in KI67_image] # MAKE THIS A SESSION STATE
# create title which is just slide 1, 2, 3, 4, ...
ER_title = [f"Slide {i+1}" for i in range(len(ER_image))] # MAKE THIS A SESSION STATE
PR_title = [f"Slide {i+1}" for i in range(len(PR_image))] # MAKE THIS A SESSION STATE
HER2_title = [f"Slide {i+1}" for i in range(len(HER2_image))] # MAKE THIS A SESSION STATE
KI67_title = [f"Slide {i+1}" for i in range(len(KI67_image))] # MAKE THIS A SESSION STATE
with st.container(border=1):
    st.write("Image Preview")
    if biomarker == "ER" and ER_image:
        carousel(create_carousel(ER_title, ER_text, ER_image))
        if st.button("Run Inference", use_container_width=True):
            with st.spinner("Running Inference..."):
                results_ER = run_inference("weights_1", ER_image)
                st.session_state.results_ER = results_ER
        st.write(st.session_state.results_ER)
    elif biomarker == "PR" and PR_image:
        carousel(create_carousel(PR_title, PR_text, PR_image))
        if st.button("Run Inference", use_container_width=True):
            with st.spinner("Running Inference..."):
                results_PR = run_inference("weights_1", PR_image)
                st.session_state.results_PR = results_PR
        st.write(st.session_state.results_PR)
    elif biomarker == "HER2" and HER2_image:
        carousel(create_carousel(HER2_title, HER2_text, HER2_image))
        if st.button("Run Inference", use_container_width=True):
            with st.spinner("Running Inference..."):
                results_HER2 = run_inference("weights_1", HER2_image)
                st.session_state.results_HER2 = results_HER2
        st.write(st.session_state.results_HER2)
    elif biomarker == "KI67" and KI67_image:
        carousel(create_carousel(KI67_title, KI67_text, KI67_image))
        if st.button("Run Inference", use_container_width=True):
            with st.spinner("Running Inference..."):
                results_KI67 = run_inference("weights_1", KI67_image)
                st.session_state.results_KI67 = results_KI67
        st.write(st.session_state.results_KI67)
# endregion: -- HOMOGEN BIOMARKER CLASSIFICATION --

# region: -- IHC SCORING MODEL --
st.subheader("IHC Scoring Model")
with st.container(border=1):
    st.write("Image Preview")
    if biomarker == "ER" and ER_image:
        carousel(create_carousel(ER_title, ER_text, ER_image), key="carousel_ER_ihc")
        if st.button("Run Inference", use_container_width=True, key="run_ihc"):
            with st.spinner("Running Inference..."):
                count_ER = []
                for i in range(len(ER_image)):
                    _, _, count_canny = canny_count(ER_image[i])
                    _, _, count_hough = hough_circle_count(ER_image[i])
                    _, _, count_watershed = watershed_detection(ER_image[i])
                    average_ER = (count_canny + count_hough + count_watershed) / 3
                    count_ER.append({"Canny": count_canny, "Hough Circle": count_hough, "Watershed": (int)(count_watershed), "Average": (int)(average_ER)})
                    st.session_state.count_ER = count_ER
        st.write(st.session_state.count_ER)
    elif biomarker == "PR" and PR_image:
        carousel(create_carousel(PR_title, PR_text, PR_image), key="carousel_PR_ihc")
        if st.button("Run Inference", use_container_width=True, key="run_ihc"):
            with st.spinner("Running Inference..."):
                count_PR = []
                for i in range(len(PR_image)):
                    _, _, count_canny = canny_count(PR_image[i])
                    _, _, count_hough = hough_circle_count(PR_image[i])
                    _, _, count_watershed = watershed_detection(PR_image[i])
                    average_PR = (count_canny + count_hough + count_watershed) / 3
                    count_PR.append({"Canny": count_canny, "Hough Circle": count_hough, "Watershed": (int)(count_watershed), "Average": (int)(average_PR)})
                    st.session_state.count_PR = count_PR
        st.write(st.session_state.count_PR)
    elif biomarker == "HER2" and HER2_image:
        carousel(create_carousel(HER2_title, HER2_text, HER2_image), key="carousel_HER2_ihc")
        if st.button("Run Inference", use_container_width=True, key="run_ihc"):
            with st.spinner("Running Inference..."):
                count_HER2 = []
                for i in range(len(HER2_image)):
                    _, _, count_canny = canny_count(HER2_image[i])
                    _, _, count_hough = hough_circle_count(HER2_image[i])
                    _, _, count_watershed = watershed_detection(HER2_image[i])
                    average_HER2 = (count_canny + count_hough + count_watershed) / 3
                    count_HER2.append({"Canny": count_canny, "Hough Circle": count_hough, "Watershed": (int)(count_watershed), "Average": (int)(average_HER2)})
                    st.session_state.count_HER2 = count_HER2
        st.write(st.session_state.count_HER2)
    elif biomarker == "KI67" and KI67_image:
        carousel(create_carousel(KI67_title, KI67_text, KI67_image), key="carousel_KI67_ihc")
        if st.button("Run Inference", use_container_width=True, key="run_ihc"):
            with st.spinner("Running Inference..."):
                count_KI67 = []
                for i in range(len(KI67_image)):
                    _, _, count_canny = canny_count(KI67_image[i])
                    _, _, count_hough = hough_circle_count(KI67_image[i])
                    _, _, count_watershed = watershed_detection(KI67_image[i])
                    average_KI67 = (count_canny + count_hough + count_watershed) / 3
                    count_KI67.append({"Canny": count_canny, "Hough Circle": count_hough, "Watershed": (int)(count_watershed), "Average": (int)(average_KI67)})
                    st.session_state.count_KI67 = count_KI67
        st.write(st.session_state.count_KI67)
# endregion: -- IHC SCORING MODEL --

# region: -- HETEROGEN ENSEMBLE MODEL --
st.subheader("Heterogen Ensemble Model")
if biomarker == "ER" and ER_image and (st.session_state.results_ER!=[]) and (st.session_state.count_ER!=[]):
    for i in range(len(ER_image)):
        st.write(f"Slide {i+1}")
        st.write(f"Homogen model output = {st.session_state.results_ER[i]['combined']}")
        st.write(f"IHC model output = {st.session_state.count_ER[i]}")
        st.write("---")
elif biomarker == "PR" and PR_image and (st.session_state.results_PR!=[]) and (st.session_state.count_PR!=[]):
    for i in range(len(PR_image)):
        st.write(f"Slide {i+1}")
        st.write(f"Homogen model output = {st.session_state.results_PR[i]['combined']}")
        st.write(f"IHC model output = {st.session_state.count_PR[i]}")
        st.write("---")
elif biomarker == "HER2" and HER2_image and (st.session_state.results_HER2!=[]) and (st.session_state.count_HER2!=[]):
    for i in range(len(HER2_image)):
        st.write(f"Slide {i+1}")
        st.write(f"Homogen model output = {st.session_state.results_HER2[i]['combined']}")
        st.write(f"IHC model output = {st.session_state.count_HER2[i]}")
        st.write("---")
elif biomarker == "KI67" and KI67_image and (st.session_state.results_KI67!=[]) and (st.session_state.count_KI67!=[]):
    for i in range(len(KI67_image)):
        st.write(f"Slide {i+1}")
        st.write(f"Homogen model output = {st.session_state.results_KI67[i]['combined']}")
        st.write(f"IHC model output = {st.session_state.count_KI67[i]}")
        st.write("---")
# endregion: -- HETEROGEN ENSEMBLE MODEL --



