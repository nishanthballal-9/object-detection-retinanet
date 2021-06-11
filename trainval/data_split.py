import json
import os
import shutil

with open('annotations/val.json', 'rt', encoding='UTF-8') as annotations:
    coco = json.load(annotations)

images = coco['images']
img_list = [img['file_name'] for img in images]

with open('annotations/train.json', 'rt', encoding='UTF-8') as annotations:
    coco = json.load(annotations)

images = coco['images']
img_list2 = [img['file_name'] for img in images]

os.mkdir('images_separated')
os.mkdir('images_separated/train')
os.mkdir('images_separated/val')

for img in img_list:
    fp = os.path.join('images', img)
    dest_fp = os.path.join('images_separated/val', img)
    shutil.copy(fp, dest_fp)

for img in img_list2:
    fp = os.path.join('images', img)
    dest_fp = os.path.join('images_separated/train', img)
    shutil.copy(fp, dest_fp)