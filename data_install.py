import kagglehub

''' 
Dataset 1 - Flickr 8K dataset

This dataset gives us two files
1. Images file of 8k images
2. Captions file

'''

# Load in dataset and get file path
flickr8k_path = kagglehub.dataset_download('adityajn105/flickr8k')

# Parse file path into images and captions
# print(flickr8k_path)
# print(flickr8k_path + r"\images")
# print(flickr8k_path + r"\captions.txt")