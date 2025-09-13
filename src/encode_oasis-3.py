import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--encoded_dir", help="Directory to save encoded dataset", type=str)
    parser.add_argument("--encoder", help="Encoder pre-trained on ImageNet", type=str)
    parser.add_argument("--numpy_dir", help="Directory to numpy dataset", type=str)
    parser.add_argument("--seed", default=42, help="Random seed (default: 42)", type=int)
    args = parser.parse_args()
    
    os.makedirs(args.encoded_dir, exist_ok=True)

    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')

    grouped_df = labels_df.groupby("Subject")["Alzheimer\'s"].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped_df["Subject"], grouped_df["Alzheimer\'s"]
    train_and_val_ids, test_ids, train_and_val_id_labels, test_id_labels = train_test_split(ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels)
    train_ids, val_ids = train_test_split(train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels)
    
    train_df = labels_df[labels_df["Subject"].isin(train_ids)]
    val_df = labels_df[labels_df["Subject"].isin(val_ids)]
    test_df = labels_df[labels_df["Subject"].isin(test_ids)]
    
    transform = torchvision.transforms.Compose([
        lambda image: torch.as_tensor(image, dtype=torch.float32),
        lambda image: image.permute(3, 0, 1, 2),
        lambda image: utils.pad_image(image),
        torchvision.transforms.Resize((224, 224)),
        lambda image: torch.rot90(image, k=1, dims=[-2, -1]),
    ])
    train_dataset = utils.Dataset(train_df, transform=transform)

    means, stds = [], []

    for image, num_slices, label in train_dataset:
        means.append(torch.mean(image, dim=(0, 2, 3)).tolist())
        stds.append(torch.std(image, dim=(0, 2, 3)).tolist())

    mean = torch.tensor(means).mean(dim=0)
    std = torch.tensor(stds).mean(dim=0)

    transform = torchvision.transforms.Compose([
        lambda image: torch.as_tensor(image, dtype=torch.float32),
        lambda image: image.permute(3, 0, 1, 2),
        lambda image: utils.pad_image(image),
        lambda image: torch.rot90(image, k=1, dims=[-2, -1]),
        torchvision.transforms.Resize((224, 224)),
        lambda image: (image - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1),
    ])

    train_dataset = utils.Dataset(train_df, transform=transform)
    val_dataset = utils.Dataset(val_df, transform=transform)
    test_dataset = utils.Dataset(test_df, transform=transform)
    
    if args.encoder == 'ViT-B/16':
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights(weights))
        model.conv_proj.weight.data = model.conv_proj.weight.data.sum(dim=1, keepdim=True)
        model.conv_proj.in_channels = 1
        model.heads = torch.nn.Identity()
    elif args.encoder == 'ConvNeXt-Tiny':
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights(weights))
        model.features[0][0].weight.data = model.features[0][0].weight.data.sum(dim=1, keepdim=True)
        model.features[0][0].in_channels = 1
        model.classifier[2] = torch.nn.Identity()
    else:
        raise NotImplementedError(f'The specified encoder \'{args.encoder}\' is not implemented.')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    X, lengths, y = [], [], []
    for image, num_slices, label in train_dataset:
        
        D, C, H, W = image.shape
        encoded_image = torch.cat([utils.encode_image(model, image[:, c:c+1]) for c in range(C)], dim=-1)
        
        X.append(encoded_image)
        lengths.append(num_slices)
        y.append(label)
        
    torch.save({
        "X": torch.cat(X),
        "lengths": tuple(lengths),
        "y": torch.stack(y),
    }, f"{args.encoded_dir}/train.pth")
        
    X, lengths, y = [], [], []
    for image, num_slices, label in val_dataset:
        
        D, C, H, W = image.shape
        encoded_image = torch.cat([utils.encode_image(model, image[:, c:c+1]) for c in range(C)], dim=-1)
        
        X.append(encoded_image)
        lengths.append(num_slices)
        y.append(label)
        
    torch.save({
        "X": torch.cat(X),
        "lengths": tuple(lengths),
        "y": torch.stack(y),
    }, f"{args.encoded_dir}/val.pth")
    
    X, lengths, y = [], [], []
    for image, num_slices, label in test_dataset:
        
        D, C, H, W = image.shape
        encoded_image = torch.cat([utils.encode_image(model, image[:, c:c+1]) for c in range(C)], dim=-1)
        
        X.append(encoded_image)
        lengths.append(num_slices)
        y.append(label)
        
    torch.save({
        "X": torch.cat(X),
        "lengths": tuple(lengths),
        "y": torch.stack(y),
    }, f"{args.encoded_dir}/test.pth")
    