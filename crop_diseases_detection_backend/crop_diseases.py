#_ _ _ _ _ _ _ _ _ _ _ _ Imaport libraires_ _ _ _ _ _ _ _ _ _ _ _ _ _ 
import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import tkinter as tk
from tkinter import filedialog

# ------------------- Config -------------------
MODEL_PATH = 'plant_disease_model.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names = None, []

#_ _ _ _ _ _ _ _Load Model_ _ _ _ _ _ _ _ _ 
try:
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    print("Checkpoint loaded successfully ✅")

    # Extract class names
    class_names = checkpoint["class_names"]
    print("Number of classes:", len(class_names))

    # Rebuild model architecture (example: ResNet18)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    #print("Model loaded successfully ✅ and ready for prediction")

except Exception as e:
    print("❌ Error loading model:", str(e))



# ------------------- Transforms -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ------------------- Prediction Functions -------------------
def predict_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top3_probs, top3_idxs = torch.topk(probs, 3)

        predictions = []
        for i in range(3):
            class_name = class_names[top3_idxs[i]]
            confidence = float(top3_probs[i])
            predictions.append({
                "name": class_name,
                "confidence": confidence
            })
    return predictions

# --- Prediction Function ---
def predict_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_idx = torch.argmax(probs).item()
        class_name = class_names[top_idx]
        confidence = float(probs[top_idx])
    return {"disease": class_name, "confidence": confidence}


# ------------------- image Upload -------------------
def upload_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    root.update()
    root.destroy()
    return file_path

# --- Upload Video ---
def upload_video():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    root.update()
    root.destroy()
    return file_path

# ------------------- Main menu -------------------
if __name__ == "__main__":
    print("\nChoose an option:")
    print("1. Upload Image and Predict")
    print("2. Upload Video and Predict")
    print("3. Live Camera Prediction")
    choice = input("Enter choice: ")

    if choice == "1":
        image_path = upload_image()
        if image_path:
            preds = predict_disease(image_path)
            print("\n--- Image Prediction ---")
            for p in preds:
                print(f"Class: {p['name']}, Confidence: {p['confidence']:.2f}")
        else:
            print("No file selected.")

    
    elif choice == "2":
        video_path = upload_video()
        if video_path:
            cap = cv2.VideoCapture(video_path)

            results = []  # store predictions

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                pred = predict_frame(frame)  # get prediction dict
                results.append(pred)

                # Show prediction on video
                display_text = f"{pred['disease']} ({pred['confidence']:.2f})"
                cv2.putText(frame, display_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Video Prediction", frame)

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            # --- Save results to CSV ---
            with open("video_predictions.csv", "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["disease", "confidence"])
                writer.writeheader()
                writer.writerows(results)

            print("Predictions saved to video_predictions.json and video_predictions.csv")
        else:
            print("No file selected.")

    elif choice == "3":
        cap = cv2.VideoCapture(0)
        print("Starting live camera... Press 'q' to quit")

        # Open CSV file for logging
        with open("live_predictions.csv", mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["frame", "disease", "confidence"])
            writer.writeheader()

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                pred = predict_frame(frame)

                # Convert dict to text for display
                text = f"{pred['disease']} ({pred['confidence']:.2f})"

                # Write prediction to CSV
                writer.writerow({
                    "frame": frame_count,
                    "disease": pred["disease"],
                    "confidence": round(pred["confidence"], 4)
                })

                # Show prediction on live feed
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Live Camera Prediction", frame)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows() 

    else:
        print("Invalid choice")

