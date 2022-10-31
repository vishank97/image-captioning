from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch 
from PIL import Image
from tqdm import tqdm
import urllib.request
import os

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}



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
    os.remove('file.jpg')
    return preds