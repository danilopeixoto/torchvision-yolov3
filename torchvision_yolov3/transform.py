import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision.models.detection.transform import resize_boxes


def resize_image(image, size):
    return F.interpolate(image[None], size, mode="bilinear")[0]


class YOLOTransform(nn.Module):
    def __init__(self, input_size, conf_thresh, nms_thresh, max_detections):
        super(YOLOTransform, self).__init__()
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections

    def forward(self, images, targets):
        image_list, target_list = [], []

        for i in range(len(images)):
            image = images[i]
            resized_image = resize_image(image, self.input_size)

            if self.training:
                target = targets[i]
                boxes = target["boxes"]
                labels = target["labels"]

                h, w = image.shape[1:]
                out_boxes = torch.zeros(boxes.shape, dtype=torch.float32, device=boxes.device)

                out_boxes[:, 0] = torch.div(boxes[:, 0] + boxes[:, 2], 2 * w)
                out_boxes[:, 1] = torch.div(boxes[:, 1] + boxes[:, 3], 2 * h)
                out_boxes[:, 2] = torch.div(boxes[:, 2] - boxes[:, 0], w)
                out_boxes[:, 3] = torch.div(boxes[:, 3] - boxes[:, 1], h)
                
                out_labels = torch.transpose(labels.unsqueeze(0) - 1, 0, 1).float()

                boxes_with_cls = torch.cat([out_boxes, out_labels], 1)
                target_list.append(boxes_with_cls)

            image_list.append(resized_image)

        out_images = torch.stack(image_list)
        out_targets = torch.stack(target_list) if self.training else None
        
        return out_images, out_targets

    def postprocess(self, predictions, image_sizes):
        preds = torch.cat(predictions, 1)
        _, max_ids = torch.max(preds[:, :, 5:], dim=2)

        detections = []

        for i in range(preds.shape[0]):
            conf_mask = preds[i, :, 4] > self.conf_thresh
            conf_ids = torch.nonzero(conf_mask).flatten()

            x = preds[i, conf_ids, 0]
            y = preds[i, conf_ids, 1]
            w = preds[i, conf_ids, 2]
            h = preds[i, conf_ids, 3]

            xmin = x - w * 0.5
            ymin = y - h * 0.5
            xmax = xmin + w
            ymax = ymin + h

            relative_boxes = torch.stack([xmin, ymin, xmax, ymax], 1)

            boxes = resize_boxes(relative_boxes, self.input_size, image_sizes[i])
            labels = max_ids[i, conf_ids].long() + 1
            scores = preds[i, conf_ids, 4]

            keep = torchvision.ops.nms(boxes, scores, self.nms_thresh)[:self.max_detections]

            detection = {
                "boxes": boxes[keep],
                "labels": labels[keep],
                "scores": scores[keep]
            }
            
            detections.append(detection)

        return detections
