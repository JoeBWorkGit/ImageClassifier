import argparse
import myutils
import json

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', action='store', default = 'flowers/test/1/image_06743.jpg', help='Path to the image,') 
parser.add_argument('--checkpoint', action='store', default = 'checkpoint.pth',  help='Directory to save')
parser.add_argument('--topk', action='store',  default = 5,  dest='topk',  help='Top 5 classes')
parser.add_argument('--cat_to_name', action='store', default = 'cat_to_name.json', dest='cat_to_name', help='File name of the mapping file')
parser.add_argument('--gpu', action='store', default='gpu', help='GPU mode on or off')

results = parser.parse_args()

image_path = results.image_path
checkpoint = results.checkpoint
topk = int(results.topk)
category_names = results.cat_to_name
gpu = results.gpu
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
model = myutils.load_checkpoint()

probs,classes = myutils.predict(image_path, model, topk)
print('results')
print(probs)
print(classes) 
print('done')
names = []
for i in classes:
    names += [cat_to_name[i]]
    
# Print name of predicted flower with highest probability
print(f"This flower is most likely to be a: '{names[0]}' with a probability of {round(probs[0]*100,4)}% ")

    