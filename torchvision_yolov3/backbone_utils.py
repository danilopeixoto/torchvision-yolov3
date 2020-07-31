from . import darknet


def darknet_backbone(backbone_name, pretrained=False, progress=True):
    '''Constructs backbone
    '''
    backbone = darknet.__dict__[backbone_name](
        pretrained = pretrained,
        progress = progress)
    
    return backbone
