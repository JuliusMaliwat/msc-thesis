import numpy as np
import torch
import torchreid
from PIL import Image
from torchvision import transforms

class Embedder:
    """
    Appearance embedder for DeepSORT BEV.
    Computes appearance embeddings from cropped images using OSNet.
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self):
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        model.to(self.device)
        model.eval()
        return model

    def _build_transform(self):
        # Same preprocessing as your original notebook
        return transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def compute(self, crops_dict):
        """
        Compute embeddings for given crops.

        Args:
            crops_dict (dict): Mapping from (frame_id, x, y) to list of cropped images (np.ndarray).

        Returns:
            list: List of embedding vectors corresponding to crops_dict.keys().
        """
        embeddings = []

        for key, crop_list in crops_dict.items():
            if not crop_list:
                # No valid crops, use zero vector
                embedding = np.zeros(512)
            else:
                crop_embeddings = []
                for crop in crop_list:
                    try:
                        emb = self.compute_single(crop)
                    except Exception as e:
                        print(f"⚠️ Error processing crop for key {key}: {e}")
                        emb = np.zeros(512)
                    crop_embeddings.append(emb)

                # Average embeddings across multiple views (cameras)
                embedding = np.mean(crop_embeddings, axis=0)

            embeddings.append(embedding)

        return embeddings

    def compute_single(self, crop):
        """
        Compute embedding for a single cropped image.

        Args:
            crop (np.ndarray): Cropped image array (H, W, C).

        Returns:
            np.ndarray: L2-normalized embedding vector (512-dim).
        """
        # Convert np.ndarray to PIL Image
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(img_tensor)
            features = features.cpu().numpy()[0]

        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()


        return features
