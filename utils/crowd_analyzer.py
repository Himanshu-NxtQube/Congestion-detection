import os
import cv2
import threading
import numpy as np
from ultralytics import YOLO

class CrowdAnalyzer:
    def __init__(self, model_path, conf_threshold, grid_rows, grid_cols, heatmap_alpha, people_threshold):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.heatmap_alpha = heatmap_alpha
        self.people_threshold = people_threshold
    
    def analyze(self, video_path, upload=False):
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        clip_count = 0
        last_frame_with_people = -9999
        current_frame_no = 0
        output_path = f"output/clip_{clip_count}.mp4"
        
        cell_w = W // self.grid_cols
        cell_h = H // self.grid_rows

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=self.conf_threshold, classes=[0], verbose=False)

            people_count = len(results[0].boxes)

            cv2.putText(frame, f"Total people: {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

            if people_count > self.people_threshold:
                last_frame_with_people = current_frame_no

            if (current_frame_no - last_frame_with_people < fps * 2):
                # If currently detecting people or within 2 seconds of the last detection
                if writer is None:
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (W, H)
                    )
                    print("New writer intialized!")
                threading.Thread(target=writer.write, args=(frame,)).start()
            else:
                if writer is not None:
                    threading.Thread(target=writer.release).start()
                    writer = None
                    print(f"Writer closed! Clip saved as clip_{clip_count}.mp4")
                    clip_count += 1

            current_frame_no += 1
            if current_frame_no % fps == 0:
                print("Current frame:", current_frame_no/fps)


        cap.release()