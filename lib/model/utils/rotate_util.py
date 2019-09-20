import torch
import PIL.Image as Image
import torchvision.transforms as transforms

def rotate_img(img, aux_label):
    if aux_label == 0:
        return img
    elif aux_label == 1:
        return img.transpose(Image.ROTATE_90)
    elif aux_label == 2:
        return img.transpose(Image.ROTATE_180)
    elif aux_label == 3:
        return img.transpose(Image.ROTATE_270)

def roate_tensor(padding_data):
    img_tensor = padding_data.type(torch.uint8)

    aux_label = int((torch.rand(1) * 4).item())

    trans = transforms.Compose([transforms.ToPILImage(),
                        transforms.Lambda(lambda img: rotate_img(img, aux_label)),
                        transforms.ToTensor()])

    img_tensor_trans = trans(img_tensor) * 255

    return img_tensor_trans.type(torch.float), aux_label

def rotate_boxes(gt_boxes, aux_label, w, h):
    gt_boxes_rotate = gt_boxes.clone()
    if aux_label == 0:
        gt_boxes_rotate = gt_boxes
    elif aux_label == 1:
        gt_boxes_rotate[:, 0] = gt_boxes[:, 1]
        gt_boxes_rotate[:, 1] = w - gt_boxes[:, 2]
        gt_boxes_rotate[:, 2] = gt_boxes[:, 3]
        gt_boxes_rotate[:, 3] = w - gt_boxes[:, 0]

    elif aux_label == 2:
        gt_boxes_rotate[:, 0] = w - gt_boxes[:, 2]
        gt_boxes_rotate[:, 1] = h - gt_boxes[:, 3]
        gt_boxes_rotate[:, 2] = w - gt_boxes[:, 0]
        gt_boxes_rotate[:, 3] = h - gt_boxes[:, 1]
    elif aux_label == 3:
        gt_boxes_rotate[:, 0] = h - gt_boxes[:, 3]
        gt_boxes_rotate[:, 1] = gt_boxes[:, 0]
        gt_boxes_rotate[:, 2] = h - gt_boxes[:, 1]
        gt_boxes_rotate[:, 3] = gt_boxes[:, 2]

    return gt_boxes_rotate