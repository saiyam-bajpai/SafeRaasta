import os
os.chdir(r"C:\Users\asus\Downloads\Pothole Detection Model RoadWatch AI")

from ultralytics import YOLO

model_path = r"C:\Users\asus\Downloads\Pothole Detection Model RoadWatch AI\outputs\best.pt"
image_path = r"C:\Users\asus\Downloads\Pothole Detection Model RoadWatch AI\testing\WhatsApp Image 2026-05-29 at 1.50.04 AM.jpeg"

model = YOLO(model_path)

results = model.predict(
    source=image_path,
    conf=0.25,
    save=True,
    project=r"C:\Users\asus\Downloads\Pothole Detection Model RoadWatch AI\runs",
    name="test_result"
)

print("Number of detections:", len(results[0].boxes))
print(f"Result saved to: C:\\Users\\asus\\Downloads\\Pothole Detection Model RoadWatch AI\\runs\\test_result\\")