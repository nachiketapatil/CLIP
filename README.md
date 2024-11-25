**CLIP Model from Scratch (Flickr30k)**

This repository contains the implementation of the CLIP (Contrastive Language-Image Pretraining) model, which was trained from scratch using the Flickr30k dataset. The model is capable of understanding the relationship between images and their corresponding textual descriptions, enabling two core functionalities:

1) Image Generation from Caption: Given an input text caption, the model generates a relevant image based on the learned relationship between the textual and visual modalities.

2) Caption Generation from Image: Given an input image, the model generates a description that best matches the content of the image.

This project is done for **FML(CS725)** at **IIT Bombay**

**Team Members**:
* Abhishek
* Nasir
* Utkarsh
* Nachiketa

**Link to PPT**:
[PPT](https://docs.google.com/presentation/d/1pLBjGjnPWlIFFBw2ThleTIKHX2gpTOmtNSQgSYOytOg/edit?usp=sharing)

**Link to CLIP repository and code**:
[CLIP Github](https://github.com/openai/CLIP)

**Link to paper**:
[CLIP Paper](https://arxiv.org/pdf/2103.00020)

**Link to Dataset**:
[Flickr30k Dataset](https://shannon.cs.illinois.edu/DenotationGraph/)

**Image used for getting captions:**

![Image used for getting captions](https://github.com/user-attachments/assets/5cdcd9e4-6b36-479b-b2eb-51801f193570)

**Response from Model:**

![Response](https://github.com/user-attachments/assets/911a4b46-426e-4031-9212-136475b131bb)

**Input given to get images : a dog playing on the grass**

**Response from Model:**

![Response](https://github.com/user-attachments/assets/128abda8-a783-4578-b4fd-cbfda97da74c)


How to run code :

Step 1 : Download CLIP.ipynb file 

Step 2 : Change paths of dataset (Image path & caption.csv file)

Step 3 : Download & add train_captions.csv, valid_captions.csv and best_model.pth to your directory
[Link to download model](https://drive.google.com/drive/folders/1ctEGkLlLWMrasLPXX6-hFVx9EeTFXuCQ?usp=sharing)


Step 4 : Run CLIP.ipynb


If you want to train the model on your system :

Step 1 : Download CLIPtrainingFinal.py

Step 2 : Change paths of dataset (Image path & caption.csv file). 
When training starts best model at each epoch will be saved

Step 3 : Add model path in CLIP.ipynb file and run.

**NOTE : You need a good GPU to train the model. We used RTX 3060 (12GB) to train & run**

