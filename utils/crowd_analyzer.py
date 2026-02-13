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
        
        cell_w = W // self.grid_cols
        cell_h = H // self.grid_rows

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ---------------- Define ROI (Middle 50%) ----------------
            x1 = W // 4
            y1 = H // 4
            x2 = x1 + (W // 2)
            y2 = y1 + (H // 2)

            roi_frame = frame[y1:y2, x1:x2]

            # ---------------- Run Detection on ROI ----------------
            results = self.model(roi_frame, conf=self.conf_threshold, classes=[0], verbose=False)

            people_count = len(results[0].boxes)

            cv2.putText(frame, f"Total people (ROI): {people_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Visualize ROI rectangle
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # ---------------- Build density grid for ROI ----------------
            density_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

            roi_w = x2 - x1
            roi_h = y2 - y1
            
            cell_w = roi_w // self.grid_cols
            cell_h = roi_h // self.grid_rows

            for box in results[0].boxes.xyxy.cpu().numpy():
                bx1, by1, bx2, by2 = map(int, box)
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2

                col = min(cx // cell_w, self.grid_cols - 1)
                row = min(cy // cell_h, self.grid_rows - 1)

                density_grid[row, col] += 1

            # ---------------- Normalize grid ----------------
            if density_grid.max() > 0:
                density_grid /= density_grid.max()

            # ---------------- Upscale to ROI size ----------------
            heatmap = cv2.resize(
                density_grid,
                (roi_w, roi_h),
                interpolation=cv2.INTER_CUBIC
            )
            heatmap = np.where(heatmap > 0.3, heatmap, 0)
            
            heatmap = np.uint8(255 * heatmap)

            # ---------------- Apply color map ----------------
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # ---------------- Overlay heatmap on ROI ----------------
            roi_overlay = cv2.addWeighted(
                heatmap_color,
                self.heatmap_alpha,
                roi_frame,
                1 - self.heatmap_alpha,
                0
            )
            
            # Place the overlay back into the main frame
            frame[y1:y2, x1:x2] = roi_overlay

            if people_count > self.people_threshold:
                last_frame_with_people = current_frame_no

            if (current_frame_no - last_frame_with_people < fps * 2):
                output_path = f"output/clip_{clip_count}.mp4"
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