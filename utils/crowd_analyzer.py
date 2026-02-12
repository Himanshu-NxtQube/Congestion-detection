import cv2
import numpy as np
from ultralytics import YOLO

class CrowdAnalyzer:
    def __init__(self, model_path, conf_threshold, grid_rows, grid_cols, heatmap_alpha):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.heatmap_alpha = heatmap_alpha
    
    def analyze(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H)
        )

        cell_w = W // self.grid_cols
        cell_h = H // self.grid_rows

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.conf_threshold, classes=[0], verbose=False)

            cv2.putText(frame, f"Total people: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ---------------- Build density grid ----------------
            density_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                col = min(cx // cell_w, self.grid_cols - 1)
                row = min(cy // cell_h, self.grid_rows - 1)

                density_grid[row, col] += 1

            # ---------------- Normalize grid ----------------
            if density_grid.max() > 0:
                density_grid /= density_grid.max()

            # ---------------- Upscale to frame size ----------------
            heatmap = cv2.resize(
                density_grid,
                (W, H),
                interpolation=cv2.INTER_CUBIC
            )
            heatmap = np.where(heatmap > 0.3, heatmap, 0)
            
            heatmap = np.uint8(255 * heatmap)

            # ---------------- Apply color map ----------------
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # ---------------- Overlay heatmap (low opacity) ----------------
            frame = cv2.addWeighted(
                heatmap_color,
                self.heatmap_alpha,
                frame,
                1 - self.heatmap_alpha,
                0
            )

            writer.write(frame)

        cap.release()
        writer.release()