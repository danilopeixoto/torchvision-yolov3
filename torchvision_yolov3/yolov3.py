import torch
import torch.nn as nn
from collections import OrderedDict

from .utils import load_state_dict_from_url
from .backbone_utils import darknet_backbone
from .transform import YOLOTransform
from .loss import YOLOLoss


__all__ = [
    "YOLOv3", "yolov3_darknet53",
]


class YOLOv3(nn.Module):
    def __init__(self,
            backbone, num_classes,
            input_size=(416, 416),
            conf_thresh=0.5,
            nms_thresh=0.1,
            max_detections=100,
            anchors=[[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]]):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections
        self.anchors = anchors
        #  transform
        self.transform = YOLOTransform(self.input_size, self.conf_thresh, self.nms_thresh, self.max_detections)
        #  backbone
        self.backbone = backbone
        _out_filters = self.backbone.layers_out_filters
        #  embedding0
        final_out_filter0 = len(self.anchors[0]) * (4 + self.num_classes)
        self.embedding0 = self._make_embedding([512, 1024], _out_filters[-1], final_out_filter0)
        #  embedding1
        final_out_filter1 = len(self.anchors[1]) * (4 + self.num_classes)
        self.embedding1_cbl = self._make_cbl(512, 256, 1)
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.embedding1 = self._make_embedding([256, 512], _out_filters[-2] + 256, final_out_filter1)
        #  embedding2
        final_out_filter2 = len(self.anchors[2]) * (4 + self.num_classes)
        self.embedding2_cbl = self._make_cbl(256, 128, 1)
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.embedding2 = self._make_embedding([128, 256], _out_filters[-3] + 128, final_out_filter2)
        #  losses
        self.losses = nn.ModuleList([YOLOLoss(self.num_classes, self.input_size, self.anchors[i]) for i in range(3)])

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def forward(self, images, targets=None):
        x, gt = self.transform(images, targets)
        
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.embedding0, x0)
        #  yolo branch 1
        x1_in = self.embedding1_cbl(out0_branch)
        x1_in = self.embedding1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x2_in = self.embedding2_cbl(out1_branch)
        x2_in = self.embedding2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)

        outputs = [out0, out1, out2]
        pred = [loss(outputs[i], gt) for i, loss in enumerate(self.losses)]
        
        if self.training:
            losses = [sum(loss) for loss in zip(*pred)]

            loss_dict = {
                "loss_box_x": losses[0],
                "loss_box_y": losses[1],
                "loss_box_width": losses[2],
                "loss_box_height": losses[3],
                "loss_objectness": losses[4],
                "loss_classifier": losses[5]
            }

            return loss_dict
        else:
            img_sizes = [img.shape[1:] for img in images]
            return self.transform.postprocess(pred, img_sizes)


model_urls = {
    "yolov3_darknet53_coco": "https://media.githubusercontent.com/media/danilopeixoto/pretrained-weights/master/yolov3_darknet53_coco.pth"
}


def yolov3_darknet53(num_classes=81, pretrained=False, pretrained_backbone=True,
        progress=True, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = darknet_backbone("darknet53", pretrained_backbone, progress)
    model = YOLOv3(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["yolov3_darknet53_coco"],
            progress=progress)
        model.load_state_dict(state_dict)
    return model
