import argparse
import myutils
import json

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', action='store', default='flowers/test/87/image_05466.jpg')
parser.add_argument('--checkpoint', action='store', default = 'checkpoint.pth',  help='Directory to save')
parser.add_argument('--topk', action='store',  default = 5,  dest='topk',  help='Top 5 classes')
parser.add_argument('--cat_to_name', action='store', default = 'cat_to_name.json', dest='cat_to_name', help='File name of the mapping file')
parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type=str)
parser.add_argument('--gpu', action='store', default='gpu', help='GPU mode on or off')

results = parser.parse_args()

image_path = results.image_path
checkpoint = results.checkpoint
topk = int(results.topk)
category_names = results.cat_to_name
arch = results.arch
gpu = results.gpu
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
model = myutils.load_checkpoint(arch)

probs,classes = myutils.predict(image_path, model, topk, gpu)
print('Results')
# print(probs)
# print(classes) 
classes_name = [cat_to_name[class_i] for class_i in classes]
print("\nFlower name (probability)")
print("=========================")
for i in range(len(probs)):
    print(f"{classes_name[i]} ({round(probs[i], 3)})")
print("")
names = []
for i in classes:
    names += [cat_to_name[i]]
    
print(f"This flower is probably a '{names[0]}' the probability is {round(probs[0]*100,4)}% ")

    