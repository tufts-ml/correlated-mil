import numpy as np
# PyTorch
import torch
import torchmetrics

def collate_fn(batch):
    images, slices, labels = zip(*batch)
    images = torch.cat(images)
    labels = torch.stack(labels)
    return images, slices, labels

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df, label_columns=["Alzheimer\'s"], transform=None):
        super().__init__()
        self.label = df[label_columns].to_numpy()
        self.path = df.path.to_numpy()
        self.transform = transform
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        data = np.load(self.path[index])
        image = data[data.files[0]]
        image = image if self.transform is None else self.transform(image)
        return image, image.shape[0], torch.as_tensor(self.label[index], dtype=torch.float32)
    
def encode_image(model, image):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()

    with torch.no_grad():
        
        if device.type == 'cuda':
            image = image.to(device)
            
        encoded_image = model(image)
        
        if device.type == 'cuda':
            encoded_image = encoded_image.cpu()
            
    return encoded_image

def evaluate(model, criterion, dataloader):

    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.eval()
    
    acc = torchmetrics.Accuracy(task="binary")
    auroc = torchmetrics.AUROC(task="binary")
    auprc = torchmetrics.AveragePrecision(task="binary")
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {"acc": 0.0, "auroc": 0.0, "auprc": 0.0, "labels": [], "logits": [], "loss": 0.0, "nll": 0.0}

    with torch.no_grad():
        for images, lengths, labels in dataloader:
            
            batch_size = len(lengths)

            if device.type == "cuda":
                images, labels = images.to(device), labels.to(device)

            params = torch.nn.utils.parameters_to_vector(model.parameters())
            logits, attn_weights = model(images, lengths)
            losses = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, params=params, N=len(dataloader.dataset))

            metrics["loss"] += (batch_size/dataset_size)*losses["loss"].item()
            metrics["nll"] += (batch_size/dataset_size)*losses["nll"].item()

            if device.type == "cuda":
                labels, logits = labels.detach().cpu(), logits.detach().cpu()

            for label, logit in zip(labels, logits):
                metrics["labels"].append(label)
                metrics["logits"].append(logit)

        labels = torch.stack(metrics["labels"]).to(torch.int)
        logits = torch.stack(metrics["logits"])
        metrics["acc"] = acc(logits, labels).item()
        metrics["auroc"] = auroc(logits, labels).item()
        metrics["auprc"] = auprc(logits, labels).item()
            
    return metrics

def generate_toy_data(N, H=768, h=1, delta=1.0, S_low=15, S_high=46, deltaS=3, p_y1=0.5, seed=42):
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    lengths = tuple(torch.randint(low=S_low, high=S_high, size=(N,), generator=g).tolist())
    X = torch.randn(sum(lengths), H, generator=g)
    u = torch.cat([torch.randint(0, length-(deltaS-1), size=(1,), generator=g) for length in lengths])
    y = torch.bernoulli(p_y1 * torch.ones(size=(N,)), generator=g).to(torch.int)
    
    for i, X_i in enumerate(torch.split(X, lengths)):
        if y[i] == 1:
            X_i[u[i]:u[i]+deltaS,h-1] += delta
            
    return X, lengths, u, y.view(-1, 1).float()

def normal_pdf(x, mu=0.0, sigma=1.0):
    
    return 1 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def pad_image(image):

    D, C, H, W = image.shape
    size = max(H, W)
    pad_val = image.min()
    
    padded_image = torch.full((D, C, size, size), pad_val, dtype=image.dtype, device=image.device)
    padded_image[:,:,(size-H)//2:(size-H)//2+H,(size-W)//2:(size-W)//2+W] = image
    
    return padded_image

def proba_y1_given_h(h, delta=1.0, deltaS=3, p_y1=0.5):
    
    S_i, = h.shape
    
    u_i = S_i - (deltaS-1)
    p_y0 = 1.0 - p_y1
    
    p_h_y0 = torch.prod(torch.tensor([normal_pdf(h[s]) for s in range(S_i)])) * p_y0
    
    p_u = (1/u_i) * torch.ones(size=(u_i,))
    p_h_u_y1 = torch.prod(torch.tensor([[normal_pdf(h[s], delta) if s >= u and s < (u + deltaS) else normal_pdf(h[s]) for u in range(u_i)] for s in range(S_i)]), dim=0)
    p_h_y1 = torch.sum(p_h_u_y1 * p_u) * p_y1
    
    p_y1_h = p_h_y1 / (p_h_y0 + p_h_y1)
    
    return p_y1_h

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None):

    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.train()
    
    acc = torchmetrics.Accuracy(task="binary")
    auroc = torchmetrics.AUROC(task="binary")
    auprc = torchmetrics.AveragePrecision(task="binary")
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {"acc": 0.0, "auroc": 0.0, "auprc": 0.0, "labels": [], "logits": [], "loss": 0.0, "nll": 0.0}

    for images, lengths, labels in dataloader:
        
        batch_size = len(lengths)

        if device.type == "cuda":
            images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        params = torch.nn.utils.parameters_to_vector(model.parameters())
        logits, attn_weights = model(images, lengths)
        losses = criterion(logits, labels, attn_weights=attn_weights, lengths=lengths, params=params, N=len(dataloader.dataset))
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        metrics["loss"] += (batch_size/dataset_size)*losses["loss"].item()
        metrics["nll"] += (batch_size/dataset_size)*losses["nll"].item()
        
        if lr_scheduler:
            lr_scheduler.step()

        if device.type == "cuda":
            labels, logits = labels.detach().cpu(), logits.detach().cpu()
        
        for label, logit in zip(labels, logits):
            metrics["labels"].append(label)
            metrics["logits"].append(logit)
            
    labels = torch.stack(metrics["labels"]).to(torch.int)
    logits = torch.stack(metrics["logits"])
    metrics["acc"] = acc(logits, labels).item()
    metrics["auroc"] = auroc(logits, labels).item()
    metrics["auprc"] = auprc(logits, labels).item()
    
    return metrics

class ToyDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, lengths, y):
        super().__init__()
        self.X = X
        self.lengths = lengths
        self.y = y

    def __len__(self):
        return len(self.lengths)
    
    def __getitem__(self, index):
        x_i = torch.split(self.X, self.lengths)[index]
        return x_i, self.lengths[index], self.y[index]
    