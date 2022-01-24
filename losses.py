import torch

eps = 1e-6


def soft_dice_loss(outputs, targets, per_image=False):
    batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image == True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images
    return losses.mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, apply_sigmoid=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.apply_sigmoid = apply_sigmoid

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return soft_dice_loss(input, target, per_image=self.per_image)


class JaccardLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)


class FocalLoss2d(torch.nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs * self.weight
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


class BCE_Jaccard(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5, num_classes=1):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

        self.jaccard_criterion = JaccardLoss(weight=weight, size_average=size_average, per_image=per_image, non_empty=non_empty, apply_sigmoid=apply_sigmoid, min_pixels=min_pixels)
        if num_classes == 1:
            self.BCE_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            self.BCE_criterion = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        return self.jaccard_criterion(input, target) + self.BCE_criterion(input, target)


def get_criterion(lossname='BCE', dataset='public_prostata_Surface', weights=1):
    num_classes = max(1, len(dataset.split('_')[2:]))

    # class_1_stat = 0.02  # this is for surface (NO CROP)
    # class_1_stat = 0.0631  # this is for surface (144x144)
    # class_1_stat = 0.00237  # this is for target (144x144)

    classes_w = torch.tensor(weights, device='cuda', dtype=torch.float)
    if lossname == 'BCE':
        if num_classes == 1:
            return torch.nn.BCEWithLogitsLoss(pos_weight=classes_w)
        else:
            return torch.nn.CrossEntropyLoss(weight=classes_w)
    elif lossname == 'focal':
        return FocalLoss2d(weight=classes_w)
    elif lossname == 'jaccard':
        return JaccardLoss(weight=classes_w, apply_sigmoid=True)
    elif lossname == 'dice':
        return DiceLoss(weight=classes_w, apply_sigmoid=True)
    elif lossname == 'BCE_jaccard':
        return BCE_Jaccard(weight=classes_w, apply_sigmoid=True, num_classes=num_classes)