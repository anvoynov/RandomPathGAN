from torch import nn
import torchvision

from blocks_prediction.nin import NetworkInNetwork


def save_hook(module, input, output):
    setattr(module, 'output', output)


class AdaptiveResNet(nn.Module):
    def __init__(self, num_classes):
        super(AdaptiveResNet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.features = self.model.avgpool
        self.features.register_forward_hook(save_hook)

        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, input):
        self.model(input)
        return self.fc(self.features.output.squeeze())


class BlocksPredictor(nn.Module):
    def __init__(self, blocks_counts, backbone):
        super(BlocksPredictor, self).__init__()
        if backbone == 'resnet18':
            backbone = AdaptiveResNet
        elif backbone == 'nin':
            backbone = NetworkInNetwork
        self.classifiers = nn.ModuleList(
            [backbone(n) for n in blocks_counts]
        )


    def forward(self, input):
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier(input))
        return predictions


def make_blocks_predictor(rp_generator, backbone='nin'):
    blocks_count = [len(bucket.blocks) for bucket in rp_generator.buckets()]
    return BlocksPredictor(blocks_count, backbone)
