import yaml
from utils.crowd_analyzer import CrowdAnalyzer

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    analyzer = CrowdAnalyzer(
        model_path=config["model"]["path"],
        conf_threshold=config["model"]["conf_threshold"],
        grid_rows=config["heatmap"]["grid_rows"],
        grid_cols=config["heatmap"]["grid_cols"],
        heatmap_alpha=config["heatmap"]["alpha"]
    )
    analyzer.analyze("videos/DJI_20250811091015_0175_D.MP4", "output/output_congestion.mp4")