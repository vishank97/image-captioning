from operator import concat
import streamlit as st
import requests
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch 
from PIL import Image
from tqdm import tqdm
import urllib.request

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



def main():
    # request_url = env_vars['components'][0]['env']['INFER_URL']
    st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
    # with open('demo/taxi.jpg','r') as image:
    # st.image('image/taxi.jpg')
    st.title("Image Caption Prediction")
    st.header('Welcome to Image Caption Prediction!')
    st.write('This is a sample app that demonstrates the prowess of ServiceFoundry ML model deployment.🚀')
    st.write('Visit the [Github](https://github.com/vishank97/new-york-taxi-fare-prediction) repo for detailed exaplaination or [Google Colab](https://colab.research.google.com/drive/1WL8cnVmqsWxh9Ok-Ml5axAuxGfnZ1A9S#scrollTo=KDVXAdh7yKei) notebook to get started right away')
    with st.form("my_form"):
        
        # is_url = st.radio("Image Type:", ("URL", "From File"))
        # if is_url == "From File":
        images = st.file_uploader("Upload Images",accept_multiple_files=True,type=["png","jpg","jpeg"])
        # else:
        #     images = st.text_input('Image URL')

        features = {
                "images": images,
                # "is_url": True if is_url == 'URL' else False
                "is_url": False
            }
            
        
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            # if is_url == 'From File':
            #     data = requests.post(url=concat(request_url, "/predict"), json=features).json()
            # else:
            # data = requests.post(url=concat(request_url, "/predict"), json=features).json()
            # if data:
            #     print(data)
            #     st.metric(label="Predicted Caption",value=data)
            # else:
            #     st.error("Error")
            st.write(predict_step(images,False))

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
    return preds

if __name__ == '__main__':
    main()