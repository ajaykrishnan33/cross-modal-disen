import json
import random

MAX_CHOICES = 20

f = open("../mscoco/annotations/captions_val2014.json", "rb")
data = json.loads(f.read())
f.close()

"""

{
    "images":[
        {
            "image_id": #####,
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
                "image_id1",
                "image_id2",
                ...,
                "image_id20"
            ],
            "answer_id": ##
        }
    ]
}

"""

img_to_ann = {}
ann_to_img = {}
for img_ann in data["annotations"]:
    img_to_ann.setdefault(img_ann["image_id"], set([]))
    img_to_ann[img_ann["image_id"]].add(img_ann["caption"])
    ann_to_img[img_ann["caption"]] = img_ann["image_id"]

captions = ann_to_img.keys()
image_ids = img_to_ann.keys()

final_data = {"images":[], "captions":[]}

for img_id in img_to_ann:
    temp = {}
    temp["image_id"] = img_id
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

    final_data["captions"].append(temp)

f = open("../mscoco/annotations/cross_modal_retrieval.json", "w")
json.dump(final_data, f, indent=4)
f.close()
