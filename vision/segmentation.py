import cv2
import numpy as np
import mediapipe as mp


class HumanSegmenter:

    def __init__(self):
        self.segmenter = None

        # Support both legacy and alternate mediapipe module layouts.
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "selfie_segmentation"):
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            return

        try:
            from mediapipe.python.solutions import selfie_segmentation

            self.segmenter = selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            self.segmenter = None

    def segment(self, image):
        """
        Generate binary silhouette mask from input image.
        """

        if self.segmenter is None:
            return self._fallback_segment(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.segmenter.process(image_rgb)

        mask = results.segmentation_mask

        if mask is None:
            raise ValueError("Segmentation failed")

        # Convert to binary mask with a lower threshold for low-contrast images
        binary_mask = (mask > 0.1).astype(np.uint8)

        # Morphological close to fill holes and gaps in clothing/torso
        # Reduced OPEN to avoid erasing thin/dark regions like the waist
        kernel = np.ones((7, 7), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return binary_mask

    def _fallback_segment(self, image):
        """
        Fallback silhouette extraction when mediapipe selfie segmentation is unavailable.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Keep the largest foreground component to approximate the human silhouette.
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Segmentation failed")

        largest = max(contours, key=cv2.contourArea)
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(binary_mask, [largest], -1, color=1, thickness=cv2.FILLED)

        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        return binary_mask
