# Common_metrics_img_gen
Common metrics for image related tasks.

### [Histogram of Oriented Gradients (HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients#:~:text=The%20histogram%20of%20oriented%20gradients,localized%20portions%20of%20an%20image.), spatial alignment between generated image and ground truth: 

![image](https://github.com/user-attachments/assets/3e2c04c2-0d50-4293-a203-d90e2bfa3d99)
  - Intro: https://medium.com/analytics-vidhya/a-gentle-introduction-into-the-histogram-of-oriented-gradients-fdee9ed8f2aa
  - Pytorch: https://gist.github.com/etienne87/b79c6b4aa0ceb2cff554c32a7079fa5a
  - Numpy: https://github.com/ahmedfgad/HOGNumPy

### IoU, boundary alignment between generated image and ground truth:

![image](https://github.com/user-attachments/assets/fb4c6398-5da0-4d03-b0f5-9e2458560e06)
  - Intro: https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
  - <details><summary>Pytorch</summary>
    
        import torch
        
        def iou (boxes1, boxes2):
            """
            计算两个边界框集合之间的交并比（IoU）。
            
            参数:
            boxes1 (torch.Tensor): 形状为 (N, 4) 的二维张量，其中 N 是边界框的数量，
            每个边界框的格式为 (x1, y1, x2, y2)，(x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标。
            boxes2 (torch.Tensor): 形状为 (M, 4) 的二维张量，其中 M 是边界框的数量，
            每个边界框的格式为 (x1, y1, x2, y2)，(x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标。
            返回:
            torch.Tensor: 形状为 (N, M) 的二维张量，其中每个元素表示 boxes1 中第 i 个边界框
            和 boxes2 中第 j 个边界框的交并比。
            """
            
            # 确保输入为二维张量
            assert boxes1.ndim == 2 and boxes1.shape[1] == 4, "boxes1 should be a 2D tensor of shape (N, 4)"
            assert boxes2.ndim == 2 and boxes2.shape[1] == 4, "boxes2 should be a 2D tensor of shape (M, 4)"
            
            # 计算交集坐标
            x_intersection = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
            y_intersection = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
            x_intersection_end = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
            y_intersection_end = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
            
            # 计算交集面积
            intersection_area = torch.clamp(x_intersection_end - x_intersection, min=0) * torch.clamp(y_intersection_end - y_intersection, min=0)
    
            # 计算并集面积
            box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
            box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
            union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - intersection_area
    
            # 计算交并比
            iou = intersection_area / union_area
            return iou
    </details>

##  MS-SSIM / SSIM, structural alignment between generated image and ground truth.
  - Pytorch: https://github.com/VainF/pytorch-msssim

## LPIPS, perceptual alignment between generated image and ground truth.
  - Python: https://github.com/richzhang/PerceptualSimilarity
