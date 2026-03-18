MISTAKES/ERRORS

## ITERATION 1

1. ### did not flatten:
 after the first iteration of writing the code - the matrix multiplication went fine, but i made a very amateur mistake. I didn't flatten the image before passing it through the encoder, and i took a single image as a (3, 32*32) vector everywhere in the script (in the epsilon generation, etc). The matrix multiplication worked - but 