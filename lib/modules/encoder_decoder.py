# Copyright (c) OpenMMLab. All rights reserved.
import torch


class EncoderDecoder(torch.nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.neck = neck

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.neck:
            x = self.neck(x)
        return x

    def forward(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self.decode_head(x)
        return out
