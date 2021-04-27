
def get_backbone(name):
    if name == "pointnet2":
        from wypr.modeling.backbone.backbone_pointnet2 import Pointnet2Backbone
        return Pointnet2Backbone
    elif name == 'sparseconv':
        raise NotImplementedError
        # from wypr.modeling.backbone.backbone_sparseconv import *
        # return None
    else:
        raise ValueError('unsupoorted backbone')