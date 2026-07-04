import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=224):
    """Augmentation cho tập Train"""
    return A.Compose([
        # 🟢 Hàm này yêu cầu "size" (Đã chuẩn)
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1),
        
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_valid_transforms(img_size=224):
    """Augmentation cho tập Valid"""
    return A.Compose([
        # 🔴 Hàm này BẮT BUỘC dùng "height" và "width". 
        # Bạn chỉ cần sửa lại dòng này là xong!
        A.Resize(height=img_size, width=img_size),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])