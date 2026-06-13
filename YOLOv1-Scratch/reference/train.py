import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from model import Yolov1
from dataset import COCODataset
from loss import YoloLoss
from utils import (
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
)

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 0
PIN_MEMORY = DEVICE == "cuda"
LOAD_MODEL = False
LOAD_MODEL_FILE = "coco_overfit.pth.tar"

NUM_CLASSES = 80
IMG_DIR = "coco_raw/val2017"
TRAIN_ANNOTATION_FILE = "coco_raw/annotations/instances_val2017.json"
TEST_ANNOTATION_FILE = "coco_raw/annotations/instances_val2017.json"

# Use small subsets first while debugging COCO.
MAX_TRAIN_IMAGES = 8
MAX_TEST_IMAGES = 8

MAP_IOU_THRESHOLD = 0.5
MAP_CONF_THRESHOLD = 0.01  
TRAIN_MAP_EVAL_MODE = False 


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        out = model(data)
        loss = loss_fn(out, targets)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def maybe_use_subset(dataset, max_images):
    if max_images is None:
        return dataset
    return Subset(dataset, range(min(max_images, len(dataset))))


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss(C=NUM_CLASSES)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=DEVICE), model, optimizer)

    train_dataset = COCODataset(
        annotation_file=TRAIN_ANNOTATION_FILE,
        img_dir=IMG_DIR,
        C=NUM_CLASSES,
        transform=transform,
        remove_empty=True,
    )

    test_dataset = COCODataset(
        annotation_file=TEST_ANNOTATION_FILE,
        img_dir=IMG_DIR,
        C=NUM_CLASSES,
        transform=transform,
        remove_empty=True,
    )

    train_dataset = maybe_use_subset(train_dataset, MAX_TRAIN_IMAGES)
    test_dataset = maybe_use_subset(test_dataset, MAX_TEST_IMAGES)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        pred_boxes, target_boxes = get_bboxes(
            train_loader,
            model,
            iou_threshold=MAP_IOU_THRESHOLD,
            threshold=MAP_CONF_THRESHOLD,
            box_format="midpoint",
            device=DEVICE,
            eval_mode=TRAIN_MAP_EVAL_MODE,
            C=NUM_CLASSES,
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes,
            target_boxes,
            iou_threshold=MAP_IOU_THRESHOLD,
            box_format="midpoint",
            num_classes=NUM_CLASSES,
        )

        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)

            import time
            time.sleep(10)


if __name__ == "__main__":
    main()
