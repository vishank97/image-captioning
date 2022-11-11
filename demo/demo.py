import streamlit as st
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch 
from PIL import Image
from tqdm import tqdm
import urllib.request
from itertools import cycle
import os

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}



with open("demo/demo.yaml", "r") as stream:
    try:
        env_vars = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# @st.cache(suppress_st_warning=True)
def show_sample_images():
    filteredImages = {'First':'sample_images/dog_sitting_in_tub.jpg','Second':'sample_images/man_holding_a_flag.jpg','Third':'sample_images/basketball_player.jpg'} 
    
    # with st.form("sample"):
    cols = cycle(st.columns(3)) 
    for filteredImage in filteredImages.values():
        next(cols).image(filteredImage, width=200,)
    for i, filteredImage in enumerate(filteredImages.values()):
        if next(cols).button("Predict Caption",key=i):
            predicted_captions = predict_step([filteredImage],False)
            st.write(str(i + 1) +'. '+ predicted_captions[0])
        # if st.button('Predict Caption',key=idx):
        #     predict_step(filteredImage,False)
        # st.write('Select images to test the caption algorithm')
        # checks = st.columns(3)
        # with checks[0]:
        #     first = st.checkbox('First')
        # with checks[1]:
        #     second = st.checkbox('Second')
        # with checks[2]:
        #     third = st.checkbox('Third')
        # submitted = st.form_submit_button("Submit")
        # if submitted:
        #     images = []
        #     if first:
        #         images.append(filteredImages['First'])
        #     if second:
        #         images.append(filteredImages['Second'])
        #     if third:
        #         images.append(filteredImages['Third'])
            # predicted_captions = predict_step(images,False)
            # for i,caption in enumerate(predicted_captions):
            #     st.write(str(i+1)+'. '+caption)
    # selected_image = st.radio('Which Image you want to select?',('First', 'Second', 'Third'),horizontal=True)

# @st.cache(suppress_st_warning=True)
def image_uploader():
    with st.form("uploader"):
        images = st.file_uploader("Upload Images",accept_multiple_files=True,type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            predicted_captions = predict_step(images,False)
            for i,caption in enumerate(predicted_captions):
                st.write(str(i+1)+'. '+caption)

def images_url():
    with st.form("url"):
        urls = st.text_input('Enter URL of Images followed by comma for multiple URLs')
        images = urls.split(',')
        submitted = st.form_submit_button("Submit")
        if submitted:
            predicted_captions = predict_step(images,True)
            for i,caption in enumerate(predicted_captions):
                st.write(str(i+1)+'. '+caption)

def main():
    st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
    st.title("Image Caption Prediction")
    st.header('Welcome to Image Caption Prediction!')
    st.write('This is a sample app that demonstrates the prowess of ServiceFoundry ML model deployment.🚀')
    st.write('Visit the [Github](https://github.com/vishank97/image-captioning) repo for detailed exaplaination and to get started right away')
    tab1, tab2, tab3 = st.tabs(["Sample Images", "Image from computer", "Image from URL"])
    with tab1:
        show_sample_images()
    with tab2:
        image_uploader()
    with tab3:
        images_url()

def predict_step(images_list,is_url):
    images = []
    for image in tqdm(images_list):
        if is_url:
            urllib.request.urlretrieve(image, "file.jpg")
            i_image = Image.open("file.jpg")
            
        else:
            i_image = Image.open(image)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    if is_url:
        os.remove('file.jpg')
    return preds

if __name__ == '__main__':
    main()