import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=80):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Layout: [classes..., conf1, x1, y1, w1, h1, conf2, x2, y2, w2, h2]
        box1_start = self.C + 1
        box2_conf = self.C + 5
        box2_start = self.C + 6

        iou_b1 = intersection_over_union(
            predictions[..., box1_start : box1_start + 4],
            target[..., box1_start : box1_start + 4],
        )
        iou_b2 = intersection_over_union(
            predictions[..., box2_start : box2_start + 4],
            target[..., box1_start : box1_start + 4],
        )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3) # 1_obj_i

        ### ==== BOX COORDINATES LOSS ==== 
        # midpoint(x,y) and size(w,h)
        # box_predictions has shape (x, y, w, h)
        box_predictions = exists_box * (
            bestbox * predictions[..., box2_start : box2_start + 4]
            + (1 - bestbox) * predictions[..., box1_start : box1_start + 4]
        ) #Check if 2nd or 1st Box is best Box
        
        box_targets = exists_box * target[..., box1_start : box1_start + 4]

        box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])

        # (N, S, S, 4) -> (N*S*S, 4): Gộp tất cả các chiều lại với nhau, ngoại trừ chiều cuối cùng
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        ### ==== OBJECT LOSS ====
        pred_box = (
            bestbox * predictions[..., box2_conf : box2_conf + 1]
            + (1 - bestbox) * predictions[..., self.C : self.C + 1]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C : self.C + 1]),
        )

        ### ==== NO OBJECT LOSS ====
        # (N, S, S, 1) -> (N, S*S)
        # No Object Box 1
        no_object_loss = self.mse(
            torch.flatten((1- exists_box) * predictions[..., self.C : self.C + 1], start_dim=1), 
            torch.flatten((1 - exists_box) * target[..., self.C : self.C + 1], start_dim=1)
        )
        # No Object Box 2
        no_object_loss += self.mse(
            torch.flatten((1- exists_box) * predictions[..., box2_conf : box2_conf + 1], start_dim=1), 
            torch.flatten((1 - exists_box) * target[..., self.C : self.C + 1], start_dim=1)
        )

        ### ==== CLASS LOSS ====
        # (N, S, S, C) -> (N*S*S, C)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        ### ==== TOTAL LOSS ====
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
