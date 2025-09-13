# PyTorch
import torch
# Importing our custom module(s)
import utils

class AttentionBasedPooling(torch.nn.Module):
    def __init__(self, in_features, temp=1.0):
        super().__init__()
        self.temp = temp
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x, lengths):
        
        attn_logits = self.mlp(x)
        attn_weights = torch.cat([torch.nn.functional.softmax(weights_i/self.temp, dim=0) for weights_i in torch.split(attn_logits, lengths)])
        attn_weighted_x = attn_weights*x
        context_vectors = torch.cat([torch.sum(attn_weighted_x_i, dim=0, keepdim=True) for attn_weighted_x_i in torch.split(attn_weighted_x, lengths)])
        
        return context_vectors, attn_weights
    
class Inflate(torch.nn.Module):
    def __init__(self, input_instances=3):
        super().__init__()
        self.input_instances = input_instances
        self.half_input_instances = int(input_instances/2)

    def forward(self, x, lengths):
        num_instances, hidden_dim = x.shape
        x = torch.cat([torch.nn.functional.pad(x_i, (0, 0, self.half_input_instances, self.half_input_instances), mode='constant', value=0.0) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + (2 * self.half_input_instances) for length in lengths)
        x = torch.cat([x_i.unfold(0, self.input_instances, 1) for x_i in torch.split(x, lengths)])
        x = x.reshape(num_instances, hidden_dim * self.input_instances)
        return x
    
class InstanceConv1d(torch.nn.Module):
    def __init__(self, in_features, kernel_size=3):
        super().__init__()
        self.in_features = in_features
        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=self.kernel_size, groups=self.in_features, padding="same")
        
    def forward(self, x, lengths):
        x = torch.cat([self.conv(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        return x
    
class MaxPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
              
        out = torch.cat([torch.max(x_i, dim=0, keepdim=True).values for x_i in torch.split(x, lengths)])
        attn_weights = None
        
        return out, attn_weights  
    
class MeanPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
              
        device = x.device        
        out = torch.cat([torch.mean(x_i, dim=0, keepdim=True) for x_i in torch.split(x, lengths)])
        attn_weights = torch.cat([torch.ones(size=(length, 1), device=device) / length for length in lengths])
        
        return out, attn_weights
    
class NormalPooling(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=128, out_features=2),
        )
        
    def get_x(self, y):
        assert y.dim() == 1, "get_x() expects 1D tensor, got shape {y.shape}"
        return torch.arange(1, len(y) + 1, device=y.device) / len(y)

    def calc_mean(self, y):
        assert y.dim() == 1, "calc_mean() expects 1D tensor, got shape {y.shape}"
        max_y = torch.max(y)
        w = torch.exp(y - max_y)
        x = self.get_x(y)
        return torch.sum(x * w) / torch.sum(w)

    def forward(self, x, lengths):
        
        means_and_stds = self.mlp(x)
        means = torch.stack([self.calc_mean(mean_i) for mean_i in torch.split(means_and_stds[:,0], lengths)])
        stds = torch.stack([torch.nn.functional.softplus(std_i.mean()) for std_i in torch.split(means_and_stds[:,1], lengths)])
        weights = torch.cat([utils.normal_pdf(self.get_x(mean_i), means[i], stds[i]) for i, mean_i in enumerate(torch.split(means_and_stds[:,0], lengths))])
        attn_weights = torch.cat([weights_i/(torch.sum(weights_i) + 1e-3) for weights_i in torch.split(weights, lengths)]).view(-1, 1)
        x = torch.stack([torch.sum(context_vector_i, dim=0) for context_vector_i in torch.split(attn_weights*x, lengths)])
        
        return x, attn_weights

class SelfAttentionPooling(torch.nn.Module):
    def __init__(self, in_features, num_heads=1, temp=1.0):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.temp = temp
        self.cls_token = torch.nn.Parameter(torch.randn(size=(1, self.in_features,)))
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=self.in_features, num_heads=self.num_heads)
    
    def forward(self, x, lengths):
        
        x = torch.cat([torch.cat((self.cls_token, x_i)) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + 1 for length in lengths)
        # First transformer layer
        out, attn_weights = zip(*[self.self_attn(x_i / self.temp**0.5, x_i / self.temp**0.5, x_i) for x_i in torch.split(x, lengths)])
        x = torch.cat(out)
        # Get attention weight values from class token
        # Remove attention weight value for class token
        attn_weights = torch.cat([attn_weights_i[0,1:] for attn_weights_i in attn_weights])        
        # Get class token
        x = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])
        return x, attn_weights
    
class PPEG(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.proj1 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=3, groups=self.in_features, padding="same")
        self.proj2 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=5, groups=self.in_features, padding="same")
        self.proj3 = torch.nn.Conv1d(self.in_features, self.in_features, kernel_size=7, groups=self.in_features, padding="same")

    def forward(self, x, lengths):
        
        # Split class tokens and features
        cls_token = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])
        x = torch.cat([x_i[1:,:] for x_i in torch.split(x, lengths)])
        lengths = tuple(length - 1 for length in lengths)
        # Reshape patch tokens into 1D image space
        # Use different sized convolutions
        proj1 = torch.cat([self.proj1(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        proj2 = torch.cat([self.proj2(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        proj3 = torch.cat([self.proj3(x_i.T.unsqueeze(0)).squeeze(0).T for x_i in torch.split(x, lengths)])
        # Fuse together different spatial information
        x = x + proj1 + proj2 + proj3
        x = torch.cat([torch.cat((cls_token[i][None,:], x_i)) for i, x_i in enumerate(torch.split(x, lengths))])

        return x

class PositionalEmbeddingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
        
        device = x.device
        positional_encoding = torch.cat([torch.arange(1, length + 1, device=device) / length for length in lengths])
        Phi = torch.cat([positional_encoding.unsqueeze(1), x], dim=1)
        
        return Phi
    
class TransformerLayer(torch.nn.Module):
    def __init__(self, in_features, num_heads=8):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.norm = torch.nn.LayerNorm(normalized_shape=self.in_features)
        self.attn = torch.nn.MultiheadAttention(embed_dim=self.in_features, num_heads=self.num_heads)
        
    def forward(self, x, lengths):
        out, attn_weights = zip(*[self.attn(x_i, x_i, x_i) for x_i in torch.split(self.norm(x), lengths)])
        x = x + torch.cat(out)
        return x, attn_weights
    
class TransformerBasedPooling(torch.nn.Module):
    def __init__(self, in_features, num_heads=8):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.cls_token = torch.nn.Parameter(torch.randn(size=(1, self.in_features,)))
        self.layer1 = TransformerLayer(in_features=in_features, num_heads=self.num_heads)
        self.position_layer = PPEG(in_features=self.in_features)        
        self.layer2 = TransformerLayer(in_features=in_features, num_heads=self.num_heads)

    def forward(self, x, lengths):
        
        device = x.device
        # Concatenate class token
        x = torch.cat([torch.cat((self.cls_token, x_i)) for x_i in torch.split(x, lengths)])
        lengths = tuple(length + 1 for length in lengths)
        # Add positional encoding
        positional_encoding = torch.cat([torch.arange(0, length, device=device) / length for length in lengths])        
        x = x + positional_encoding[:,None]
        # First transformer layer
        x, _ = self.layer1(x, lengths)
        # Pyramid position encoding generator layer
        x = self.position_layer(x, lengths)
        # Second transformer layer
        x, attn_weights = self.layer2(x, lengths)
        # Get attention weight values from class token
        # Remove attention weight value for class token
        attn_weights = torch.cat([attn_weights_i[0,1:] for attn_weights_i in attn_weights])
        # Get class token
        x = torch.stack([x_i[0,:] for x_i in torch.split(x, lengths)])
        
        return x, attn_weights
