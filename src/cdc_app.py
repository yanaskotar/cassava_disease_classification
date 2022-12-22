import streamlit as st
import pandas as pd
import numpy as np
import torch
import cv2
import os
import albumentations as A
from albumentations import pytorch as ATorch

import simple_model as simple_model

def get_transforms():
        return A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    p=1.0
                ),
                ATorch.transforms.ToTensorV2(p=1.0),
            ],
            p=1.0
        )

if __name__=='__main__':
    model = simple_model.EfficientNetModel()
    weights = torch.load('resources/models/best_model.torch', map_location=torch.device('cpu'))["model_state_dict"]
    model.load_state_dict(weights)
    model.eval().to('cpu')

    label_map = pd.read_json("resources/data/label_num_to_disease_map.json", typ="series")

    with st.form('Form'):
        image_file = st.file_uploader('Give me a picture of a cassava plant', type=['png', 'jpg'])
        submitted = st.form_submit_button('submit')
        
        if submitted:
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image = get_transforms()(image=image)['image']
            image = image.reshape(1, 3, 224, 224)
            
            with torch.no_grad():
                output = torch.nn.functional.softmax(model(image), dim=1)
                prediction = np.argmax(output.to('cpu'))

            st.image(image_file)
            st.write(f"Diagnosis: {prediction}, {label_map[int(prediction)]}")
            st.write(f"Score: {output.max():.3f}")


