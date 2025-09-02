from vip import load_vip
from torchvision import transforms as T
from PIL import Image
import torch

def cauculate():
    goal_imgs = []
    goal_image_paths = ["/home/kkk/workspace/hil-serl/photos/photo15.png",]
                        # "/home/kkk/workspace/hil-serl/photos/photo1.png",
                        # "/home/kkk/wprkspace/hil-serl/photos/photo2.png",
                        # "/home/kkk/wprkspace/hil-serl/photos/photo3.png",
                        # "/home/kkk/wprkspace/hil-serl/photos/photo4.png"]
    image_path = "/home/kkk/workspace/hil-serl/photos/photo19.png"
    image = Image.open(image_path).convert("RGB")
    for img_path in goal_image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            print(f"已加载目标图片: {img_path}")
            goal_imgs.append(img)
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")
    print(f"共加载了 {len(goal_imgs)} 张目标图片")
    model = load_vip()
    model.to("cuda")
    model.eval()
    transform = T.Compose([T.Resize(224),T.ToTensor()])
    
    array = [*goal_imgs, image]
    image_array = torch.stack([transform(i) for i in array])
    
    with torch.no_grad():
        embeddings = model(image_array)
        
    distance = (embeddings[-1] - embeddings[:-1]).norm(dim=1).min().item()
    print(f"distance={distance}")
    
def main():
    cauculate()

if __name__ == "__main__":
    main()