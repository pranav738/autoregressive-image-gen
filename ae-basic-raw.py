import time
import torch
import torchvision

data_root = '../data'

device = torch.device("cuda:0")
print('Device:', device)


random_seed = 123
learning_rate = 0.005
num_epochs = 5
batch_size = 256

num_features = 784 
num_hidden_1 = 32




train_dataset = torchvision.datasets.MNIST(root = data_root, train = True, transform = torchvision.transforms.ToTensor(), download = True)

test_dataset = torchvision.datasets.MNIST(root = data_root, train = False, transform = torchvision.transforms.ToTensor(), download = True) 

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True) 
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False) 



'''
model
'''
class Autoencoder(torch.nn.Module):

    def __init__(self, num_features):
        super(Autoencoder,self).__init__() 
        #ENCODER
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1) 
        ##Decoder
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_features)



    def forward(self, x):        
        ##ENCODER
        encoded = self.linear_1(x) 
        encoded = torch.nn.functional.leaky_relu(encoded)

        ##DECODER 
        logits = self.linear_2(encoded) #is this a method of linear_2? Why isn't it a dotmethod then, like linear_2.output(encoded)? That makes way more sense. we wouldn't even have to assign the linear layer to a variable in that case. but then, there would be no point of creating the object because we wouldn't be able to store the weights anyway, right? 

        decoded = torch.sigmoid(logits) #'logits' is the new output tensor, the reconstructed one right? So why are we performing sigmoid on it? And why is it a direct torch.sigmoid, shouldn't that be a 'functional', like torch.nn.functional.sigmoid()?

        return decoded 



torch.manual_seed(random_seed) #again with the non-dot method. why use a torch.manual_seed here, can't we just use the seed we already made? And why a 'manual_seed', is there an automatic seed? 

model = Autoencoder(num_features = num_features) #remember - this is essentially the number of neurons in the first layer, which we need to pass in as an argument because it 'defines' the network architecture.

model = model.to(device) #we store the model in the gpu already?

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) #what does model.parameters() pass as an argument? I need to also check what the optimizer object stores, i forgot




'''
TRAINING LOOP
'''


start_time = time.time() #bruh. and why are we storing this again? 

for epoch in range(num_epochs):
    for batch_idx, (features,targets) in enumerate(train_loader): #you can iterate over the train_loader?

        features = features.view(-1, 28*28).to(device) #what are the tensor sizes of this? why 28*28? isn't that 784 (the input size); but the image size is 768 pixels right? where did the other 16 elements come from? the cls token and shit? 

        decoded = model(features) #okay i got it. so since we're using a batch size, the -1 in the features.view corresponds to the batch size?

        cost = torch.nn.functional.binary_cross_entropy(decoded, features)

        optimizer.zero_grad() #why? 
        cost.backward()

        #update the model parameters
        optimizer.step() #when we passed the mdoel params to this, did we do it by reference? so this can directly update those params? 


        # logging (this is rudimentary, i need to use wandb too later):

        if not batch_idx % 50: #huhh?? what's this conditional mean? 
            print('epoch: %03d/%03d | batch %03d/%03d | cost: %.4f' %(epoch+1, num_epochs, batch_idx, len(train_loader), cost))

    print('time elapsed: %.3f min' %((time.time() - start_time)/60))

print('Total Training Time: %.3f min' %((time.time() - start_time)/60))




