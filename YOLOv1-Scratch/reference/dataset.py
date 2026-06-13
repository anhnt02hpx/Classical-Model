import json
import os
from collections import defaultdict

import torch
from PIL import Image


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        annotation_file,
        img_dir,
        S=7,
        B=2,
        C=80,
        transform=None,
        remove_empty=False,
    ):
        with open(annotation_file, "r") as f:
            coco = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        # COCO category_id is not continuous, so map it to 0..79 for YOLO.
        self.category_id_to_class = {
            category["id"]: idx for idx, category in enumerate(coco["categories"])
        }

        self.images = coco["images"]
        self.boxes_by_image_id = defaultdict(list)

        for annotation in coco["annotations"]:
            if annotation.get("iscrowd", 0) == 1:
                continue

            image_id = annotation["image_id"]
            class_label = self.category_id_to_class[annotation["category_id"]]
            x_min, y_min, width, height = annotation["bbox"]

            if width <= 0 or height <= 0:
                continue

            self.boxes_by_image_id[image_id].append(
                [class_label, x_min, y_min, width, height]
            )

        if remove_empty:
            self.images = [
                image for image in self.images if len(self.boxes_by_image_id[image["id"]]) > 0
            ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_id = image_info["id"]
        img_path = os.path.join(self.img_dir, image_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        boxes = []
        image_width = image_info["width"]
        image_height = image_info["height"]

        for box in self.boxes_by_image_id[image_id]:
            class_label, x_min, y_min, width, height = box

            # COCO: [x_min, y_min, width, height] in pixels.
            # YOLO: [class, x_center, y_center, width, height] normalized 0..1.
            x = (x_min + width / 2) / image_width
            y = (y_min + height / 2) / image_height
            width = width / image_width
            height = height / image_height

            x = min(max(x, 0), 1)
            y = min(max(y, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)

            boxes.append([class_label, x, y, width, height])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            i, j = min(i, self.S - 1), min(j, self.S - 1)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                label_matrix[i, j, self.C + 1 : self.C + 5] = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
