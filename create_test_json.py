import json
import random
import numpy as np

MAX_CHOICES = 20

size = 0.1

captions_file = open("../temp/coco_dev_caps.txt", "r")
captions = [x.strip() for x in captions_file]
captions_file.close()

images = np.load("../temp/coco_dev_ims.npy")

metadata_file = open("../temp/coco_val.txt", "r")
metadata = [x.strip() for x in metadata_file]
metadata_file.close()

"""

{
    "metadata":[
    ],
    "images":[
        {
            "image_id": #####,
            "image": [###],
            "choice_list": [
                "<caption1>",
                "<caption2>",
                "<caption3>",
                ...
                "<caption20>"
            ],
            "answer_id": ##
        }
    ],
    "captions":[
        {
            "caption": "sksfjsf sdfsf s sdff sf sdf sdf ",
            "choice_list": [
                [###],
                [###],
                ...,
                [###]
            ],
            "answer_id": ##
        }
    ]
}

"""

final_data = {"images":[], "captions":[]}

# final_data["metadata"] = {}

# for img in data["images"]:
#     final_data["metadata"][img["id"]] = img["file_name"]

img_to_ann = {}
ann_to_img = {}
id_to_data = {}
for i, file_name in enumerate(metadata):
    img_to_ann.setdefault(file_name, set([]))
    img_to_ann[file_name].add(captions[i])
    ann_to_img[captions[i]] = file_name
    id_to_data[file_name] = images[i]

for img_id in metadata:
    temp = {}
    temp["image_id"] = img_id
    temp["image"] = id_to_data[img_id]
    temp["choice_list"] = set([])
    answer = random.choice(list(img_to_ann[img_id]))
    temp["choice_list"].add(answer)

    while len(temp["choice_list"]) < MAX_CHOICES:
        choice = random.choice(captions)
        if choice in img_to_ann[img_id] or choice in temp["choice_list"]:
            continue
        temp["choice_list"].add(choice)

    temp["choice_list"] = list(temp["choice_list"])
    random.shuffle(temp["choice_list"])

    temp["answer_id"] = temp["choice_list"].index(answer)

    final_data["images"].append(temp)

for caption in captions:
    temp = {}
    temp["caption"] = caption
    temp["choice_list"] = set([])
    answer = ann_to_img[caption]
    temp["choice_list"].add(answer)

    while len(temp["choice_list"]) < MAX_CHOICES:
        choice = random.choice(image_ids)
        temp["choice_list"].add(choice)

    temp["choice_list"] = list(temp["choice_list"])
    random.shuffle(temp["choice_list"])

    temp["answer_id"] = temp["choice_list"].index(answer)

    choice_list = []

    for img_id in temp["choice_list"]:
        choice_list.append(id_to_data[img_id])

    temp["choice_list"] = choice_list

    final_data["captions"].append(temp)

final_data["images"] = final_data["images"][:int(len(final_data["images"])*size)]
final_data["captions"] = final_data["captions"][:int(len(final_data["captions"])*size)]

f = open("../temp/cross_modal_retrieval.pkl", "wb")
import pickle
pickle.dump(final_data, f)
f.close()
