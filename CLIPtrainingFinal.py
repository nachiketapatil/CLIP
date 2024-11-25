import os
import cv2
import timm
import torch
import logging
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from albumentations import Compose, Resize, Normalize
from torch.utils.data import DataLoader

# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("training_logs.txt")
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configurations
class Configurations:
    IMAGE_DIR = "/home/guest1/nachiketa/fmlproj/new/1/flickr30k_images/flickr30k_images"
    CAPTIONS_FILE = "/home/guest1/nachiketa/fmlproj/new/1/flickr30k_images/results.csv"
    BATCH_SIZE = 64
    WORKERS = 4
    IMG_LR = 1e-6
    TXT_LR = 1e-7
    PROJ_LR = 1e-5
    WEIGHT_DECAY = 1e-7
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_MODEL_NAME = "vit_base_patch32_224"
    TXT_MODEL_NAME = "distilbert-base-uncased"
    IMG_SIZE = 224
    MAX_TOKEN_LEN = 200
    TEMP_FACTOR = 1.0
    PROJECTION_DIM = 256

# Model Components
class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = timm.create_model(Configurations.IMG_MODEL_NAME, pretrained=True, num_classes=0)
        self.text_encoder = DistilBertModel.from_pretrained(Configurations.TXT_MODEL_NAME)
        self.image_projection = nn.Linear(768, Configurations.PROJECTION_DIM)
        self.text_projection = nn.Linear(768, Configurations.PROJECTION_DIM)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        image_embeddings = F.normalize(self.image_projection(image_features), dim=1)

        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_embeddings = F.normalize(self.text_projection(text_features), dim=1)

        return image_embeddings, text_embeddings

# Dataset and DataLoader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, captions, tokenizer, transforms):
        self.image_paths = image_paths.tolist()
        self.captions = captions.tolist()
        self.tokenizer = tokenizer
        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = os.path.join(Configurations.IMAGE_DIR, self.image_paths[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]

        tokenized_captions = self.tokenizer(
            self.captions[idx], padding="max_length", truncation=True,
            max_length=Configurations.MAX_TOKEN_LEN, return_tensors="pt"
        )
        return {
            "image": torch.tensor(image).permute(2, 0, 1).float(),
            "input_ids": tokenized_captions["input_ids"].squeeze(0),
            "attention_mask": tokenized_captions["attention_mask"].squeeze(0),
        }

    def __len__(self):
        return len(self.captions)

def get_data_transforms():
    return Compose([
        Resize(Configurations.IMG_SIZE, Configurations.IMG_SIZE, always_apply=True),
        Normalize(max_pixel_value=255.0, always_apply=True),
    ])

# Training and Validation Loops
def train_one_epoch(model, loader, optimizer):
    model.train()
    loss_tracker = 0.0
    for batch in tqdm(loader):
        images = batch["image"].to(Configurations.DEVICE)
        input_ids = batch["input_ids"].to(Configurations.DEVICE)
        attention_mask = batch["attention_mask"].to(Configurations.DEVICE)

        image_embeddings, text_embeddings = model(images, input_ids, attention_mask)
        similarity_matrix = text_embeddings @ image_embeddings.T / Configurations.TEMP_FACTOR
        labels = torch.arange(similarity_matrix.size(0)).to(Configurations.DEVICE)

        loss = F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.T, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tracker += loss.item()

    return loss_tracker / len(loader)

def validate_one_epoch(model, loader):
    model.eval()
    loss_tracker = 0.0
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(Configurations.DEVICE)
            input_ids = batch["input_ids"].to(Configurations.DEVICE)
            attention_mask = batch["attention_mask"].to(Configurations.DEVICE)

            image_embeddings, text_embeddings = model(images, input_ids, attention_mask)
            similarity_matrix = text_embeddings @ image_embeddings.T / Configurations.TEMP_FACTOR
            labels = torch.arange(similarity_matrix.size(0)).to(Configurations.DEVICE)

            loss = F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.T, labels)
            loss_tracker += loss.item()

    return loss_tracker / len(loader)

if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained(Configurations.TXT_MODEL_NAME)
    df = pd.read_csv(Configurations.CAPTIONS_FILE, delimiter="|")
    df.columns = ["image", "caption_number", "caption"]
    df["caption"] = df["caption"].fillna("Missing caption").astype(str).str.strip()

    train_images, val_images, train_captions, val_captions = train_test_split(
        df["image"], df["caption"], test_size=0.2, random_state=42
    )

    train_df = pd.DataFrame({"image": train_images, "caption": train_captions})
    valid_df = pd.DataFrame({"image": val_images, "caption": val_captions})
    train_df.to_csv("train_captions.csv", index=False)
    valid_df.to_csv("valid_captions.csv", index=False)
    logger.info("Saved captions: train_captions.csv and valid_captions.csv")

    transforms = get_data_transforms()
    train_dataset = CustomDataset(train_images, train_captions, tokenizer, transforms)
    val_dataset = CustomDataset(val_images, val_captions, tokenizer, transforms)

    train_loader = DataLoader(train_dataset, batch_size=Configurations.BATCH_SIZE, shuffle=True, num_workers=Configurations.WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Configurations.BATCH_SIZE, shuffle=False, num_workers=Configurations.WORKERS)

    model = CLIPModel().to(Configurations.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Configurations.IMG_LR, weight_decay=Configurations.WEIGHT_DECAY)

    best_loss = float("inf")
    for epoch in range(Configurations.EPOCHS):
        logger.info(f"Epoch {epoch+1}/{Configurations.EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = validate_one_epoch(model, val_loader)

        logger.info(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.info("Saved best model as best_model.pth")
