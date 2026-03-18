import torch
import torch.nn.functional as F
from torchvision import datasets, transforms 
import torch.utils.data
from torch.utils.data import DataLoader, Subset
import time

#data root
data_root = '../data'
#device settings 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#architecture (this is for cifar100)
num_features = 1024
num_hidden_1 = 512 # (setting arbitrarily for now)
num_latent = 32 #again, setting arbitrarily for now). this is good enough that 
#hyperparams
seed = 123
learning_rate = 0.0005 #sweep - what happens? I heard the 'decoder learns faster', so what do i infer from the lr? What problems should be troubleshooted by fixing lr?
batch_size = 256 #arbitrarily; small-sized images, even 512 should be fine for CIFAR10.
num_epochs = 50 #this is way more than the standard autoencoder (non-vae) on mnist. why? 

torch.manual_seed(seed)
generator = torch.Generator().manual_seed(123)

full_training_set = datasets.CIFAR10(root=data_root, train=True, transform=transforms.ToTensor(), download=True)

training_set, val_set = torch.utils.data.random_split(full_training_set, [45000,5000], generator=generator)

#mean/std computation for training set.

mean=0.0
std=0.0
total=0
mean_comp_loader = DataLoader(dataset=full_training_set, batch_size=512, shuffle=False)

for x,y in mean_comp_loader:
    current_batch_size = x.size(0)
    total += current_batch_size
    x = x.view(current_batch_size, x.size(1), -1)    #x = (512,3,1024)
    mean += x.mean(2).sum(0)    #x.mean(2) = (512,3). sum overa ll images in the bathc to get 3-element tensor - in which each element is the sum of all R means or G means or B means. then divide later by total no of images
    std += x.std(2).sum(0)
    
mean /= total 
std /= total 


#now we computed everything. now we create NEW transforms.Compose object, with the normalization one as well. then we reload the datset.

new_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])

full_training_set = datasets.CIFAR10(root = data_root, train=True,transform=new_transforms, download=False) #download is false why? transforms changed no? anyways true wouldnt change anything no?

train_set = Subset(full_training_set, training_set.indices)
val_set = Subset(full_training_set, val_set.indices)
test_set = datasets.CIFAR10(root=data_root, train=False, transform=new_transforms, download=True)

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
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(device) #size is batchsize*3; essentially, for a particular image we train on, we get an RGB random tensor.
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

    
model = VariationalAutoencoder(num_features, num_hidden_1, num_latent)
model = model.to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



#training loooooooooooooooooooooooooooooooooooooop


start_time = time.time()

for epoch in range(num_epochs):
    for batch_idx, (features,targets) in enumerate(train_loader):
        
        features = features.view(batch_size ,3 , 32*32).to(device) #(batchsize, 3, 1024)

        z_mean, z_log_var, encoded, decoded = model(features)

        #cost
        kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()
        pixelwise_mse = F.mse(decoded, features)
        alpha = 0.5
        cost = (alpha)*pixelwise_mse + (1-alpha)*kl_divergence

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

