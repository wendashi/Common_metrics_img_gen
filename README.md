# Common_metrics_img_gen
A collection of common metrics for image related tasks.


### CLIP /SigLIP score:
  - For image-text (prompt) alignment.
  - CLIP score (Diffusers): https://huggingface.co/docs/diffusers/en/conceptual/evaluation#text-guided-image-generation
  - CLIP score (Pytorch): https://github.com/Taited/clip-score
  - SigLIP (Transformers): https://huggingface.co/docs/transformers/en/model_doc/siglip

##  MS-SSIM / SSIM:
  - For structural alignment between generated image and ground truth.

![image](https://github.com/user-attachments/assets/5202565f-46ce-485b-9a3a-577d2328f40f)
  - Intro: https://research.nvidia.com/publication/2020-07_Understanding-SSIM
  - Pytorch: https://github.com/VainF/pytorch-msssim

## LPIPS:
  - For perceptual alignment between generated image and ground truth.
  - Python: https://github.com/richzhang/PerceptualSimilarity
    
### [Histogram of Oriented Gradients (HOG)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients#:~:text=The%20histogram%20of%20oriented%20gradients,localized%20portions%20of%20an%20image.): 
  - For spatial alignment between generated image and ground truth.
    
![image](https://github.com/user-attachments/assets/3e2c04c2-0d50-4293-a203-d90e2bfa3d99)
  - Intro: https://medium.com/analytics-vidhya/a-gentle-introduction-into-the-histogram-of-oriented-gradients-fdee9ed8f2aa
  - Pytorch: https://gist.github.com/etienne87/b79c6b4aa0ceb2cff554c32a7079fa5a
  - Numpy: https://github.com/ahmedfgad/HOGNumPy

### IoU:
  - For boundary (location) alignment between generated image and ground truth.
    
![image](https://github.com/user-attachments/assets/fb4c6398-5da0-4d03-b0f5-9e2458560e06)
  - Intro: https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef
  - <details><summary>Pytorch</summary>
    
        import torch

        def iou(boxes1, boxes2):
            """
            Calculate the Intersection over Union (IoU) between two sets of bounding boxes.
        
            Parameters:
            boxes1 (torch.Tensor): A two - dimensional tensor of shape (N, 4), where N is the number of bounding boxes.
            Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the coordinate of the top - left corner, and (x2, y2) is the coordinate of the bottom - right corner.
            boxes2 (torch.Tensor): A two - dimensional tensor of shape (M, 4), where M is the number of bounding boxes.
            Each bounding box is in the format of (x1, y1, x2, y2), where (x1, y1) is the coordinate of the top - left corner, and (x2, y2) is the coordinate of the bottom - right corner.
            Returns:
            torch.Tensor: A two - dimensional tensor of shape (N, M), where each element represents the IoU of the i - th bounding box in boxes1 and the j - th bounding box in boxes2.
            """
        
            # Ensure the inputs are two - dimensional tensors
            assert boxes1.ndim == 2 and boxes1.shape[1] == 4, "boxes1 should be a 2D tensor of shape (N, 4)"
            assert boxes2.ndim == 2 and boxes2.shape[1] == 4, "boxes2 should be a 2D tensor of shape (M, 4)"
        
            # Calculate the coordinates of the intersection
            x_intersection = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
            y_intersection = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
            x_intersection_end = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
            y_intersection_end = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
        
            # Calculate the area of the intersection
            intersection_area = torch.clamp(x_intersection_end - x_intersection, min=0) * torch.clamp(y_intersection_end - y_intersection, min=0)
        
            # Calculate the area of the union
            box1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
            box2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
            union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - intersection_area
        
            # Calculate the IoU
            iou = intersection_area / union_area
            return iou
    </details>
