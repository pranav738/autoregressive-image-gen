import torch
import torch.nn.functional as F
from torchvision import datasets, transforms 
import torch.utils.data
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import sys #used this in the model shape test just after the model class
import wandb

#data root
data_root = '../../data'
#device settings 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#architecture (this is for cifar100)
num_features = 3072 #channel * 32^2
num_hidden_1 = 512 # (setting arbitrarily for now)
num_latent = 32 #again, setting arbitrarily for now). this is good enough that 
#hyperparams
seed = 123
learning_rate = 0.0005 #sweep - what happens? I heard the 'decoder learns faster', so what do i infer from the lr? What problems should be troubleshooted by fixing lr?
batch_size = 256 #arbitrarily; small-sized images, even 512 should be fine for CIFAR10.
num_epochs = 50 #this is way more than the standard autoencoder (non-vae) on mnist. why?
use_kl_annealing = False 
anneal_epochs = 20

torch.manual_seed(seed)
generator = torch.Generator().manual_seed(123)

wandb.init(
    project="cifar10-vae-sweeps",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "use_kl_annealing": use_kl_annealing,
        "anneal_epochs": anneal_epochs
    }
)







full_training_set = datasets.CIFAR10(root = data_root, train=True,transform=transforms.ToTensor(), download=False) #download is false why? transforms changed no? anyways true wouldnt change anything no?

training_indices, val_indices = torch.utils.data.random_split(range(len(full_training_set)), [45000,5000], generator = generator)

train_set = Subset(full_training_set, training_indices.indices) #we're using the exact same indices as our original training set choice. this is important
val_set = Subset(full_training_set, val_indices.indices)
test_set = datasets.CIFAR10(root=data_root, train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle = False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle = False)




##########################
### MODEL
##########################

class VariationalAutoencoder(torch.nn.Module):

    def __init__(self, num_features, num_hidden_1, num_latent):
        super(VariationalAutoencoder, self).__init__()
        
        ### encoder layers
        self.hidden_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.z_mean = torch.nn.Linear(num_hidden_1, num_latent)
        self.z_log_var = torch.nn.Linear(num_hidden_1, num_latent)
        
        
        ### decoder layers
        self.linear_3 = torch.nn.Linear(num_latent, num_hidden_1)
        self.linear_4 = torch.nn.Linear(num_hidden_1, num_features)

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn_like(z_mu).to(device)    #z_mu.size() is (batch_size, num_latent). #WHAT ABOUT THE GENERATOR??
        #initially used torch.randn(z_mu.size(0), z_mu.size(1) for this, similar to raschka's tutorial. but randn_like(z_mu) is cleaner.)
        # note that log(x^2) = 2*log(x); hence divide by 2 to get std_dev
        # i.e., std_dev = exp(log(std_dev^2)/2) = exp(log(var)/2)
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def encoder(self, features):
        x = self.hidden_1(features)
        x = F.leaky_relu(x, negative_slope=0.0001)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded
    
    def decoder(self, encoded):
        x = self.linear_3(encoded)
        x = F.leaky_relu(x, negative_slope=0.0001)
        x = self.linear_4(x)
        decoded = torch.sigmoid(x)
        return decoded

    def forward(self, features):
        
        z_mean, z_log_var, encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        
        return z_mean, z_log_var, encoded, decoded


##model shape test. added in the second iteration. 
#model = VariationalAutoencoder(3072, 512, 32).to(device)
#dummy_batch = torch.randn(5, 3072).to(device) # Batch of 5 flat images
#z_mean, z_log_var, encoded, decoded = model(dummy_batch)
#print("Decoded shape:", decoded.shape) # Should be [5, 3072]
#sys.exit() # Stop the script here so it doesn't run the broken training loop


    
model = VariationalAutoencoder(num_features, num_hidden_1, num_latent)
model = model.to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



#training loooooooooooooooooooooooooooooooooooooop


start_time = time.time()

for epoch in range(num_epochs):
    training_set_recon = 0.0
    training_set_kl = 0.0
    for batch_idx, (features,targets) in enumerate(train_loader):
        
        features = features.view(features.size(0), 3072).to(device) #(batchsize, 3072)

        z_mean, z_log_var, encoded, decoded = model(features)

        #cost
        minibatch_train_kl = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        minibatch_train_recon = F.mse_loss(decoded, features, reduction='sum')

        if use_kl_annealing: 
            kl_weight = min(1.0, (epoch+1)/anneal_epochs)
        else:
            kl_weight = 1.0

        minibatch_cost = (minibatch_train_recon + (kl_weight)*minibatch_train_kl)/features.size(0)
        training_set_kl += minibatch_train_kl
        training_set_recon += minibatch_train_recon

        optimizer.zero_grad()
        minibatch_cost.backward()
        optimizer.step()

    training_set_recon /= len(train_set)
    training_set_kl /= len(train_set )
    training_set_cost = training_set_recon + training_set_kl
    wandb.log({
        "epoch": epoch,
        "total_loss": training_set_cost.item(), #this is only the fucking training loop and the training loss.
        "train_recon_mse": training_set_recon.item(),
        "train_kl_divergence": training_set_kl.item(),
        "kl_weight": kl_weight
    })


    model.eval()
    with torch.no_grad():
        val_set_recon = 0.0
        val_set_kl = 0.0

        for features, targets in val_loader:
            features = features.view(features.size(0), 3072).to(device)
            z_mean, z_log_var, encoded, decoded = model(features)

            minibatch_val_kl = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
            minibatch_val_recon = F.mse_loss(decoded, features, reduction='sum')
            val_set_recon += minibatch_val_recon
            val_set_kl += minibatch_val_kl

        val_set_recon /= len(val_set)
        val_set_kl /= len(val_set)

        wandb.log({"val_recon_mse": val_set_recon, "val_kl_divergence": val_set_kl})

        #the below code is to reconstruct 8 random images from the val set. i'm putting it under the same torch.no_grad().

        sample_features, _ = next(iter(val_loader))
        sample_features = sample_features[:8].view(8,3072).to(device)
        _, _, _, decoded = model(sample_features)
        decoded = decoded.view(8, 3, 32, 32)
        grid = torch.cat([sample_features.view(8,3,32,32), decoded], dim = 0)
        wandb.log({"reconstructions":wandb.Image(grid, caption="Top row - original, bottom row - recon. image")})
