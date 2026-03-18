MISTAKES/ERRORS

## ITERATION 1

1. ### did not flatten:
 after the first iteration of writing the code - the matrix multiplication went fine, but i made a very amateur mistake. I didn't flatten the image before passing it through the encoder, and i took a single image as a (3, 32*32) vector everywhere in the script (in the epsilon generation, etc). The matrix multiplication worked - but the latent space was effectively of 3x32 = 96 dimensions; since each channel was considered completely different. so anyt correlations between different colours of the same pixel were ignored, all were considered independent. 

 2. ### training loop loss error: 
 When taking the reconstruction loss, I initially took the mse without the 'reduction' argument set to the string 'sum'. So i was actually computing the MSE over the batch, while the KL term was the sum over the entire batch - the two were on completely different scales. so the KL term was dominating heavily; which wouldn't work out with good results if i used the standard weight of 1 for both

 note: another error (which i tried to fix from the above) to take the per-pixel SSE over all the images in a batch - because technically that's what the recon loss respresents. but the same problem of scaling occurs here as well - if i multiply a per-image sse with 3072, the scale will be way different from the KL term. 

