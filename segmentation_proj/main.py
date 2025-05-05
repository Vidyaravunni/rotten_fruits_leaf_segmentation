import cv2
import numpy as np
import matplotlib.pyplot as plt

# === PARAMETERS ===
lower = np.array([10, 50, 20])    # HSV lower bound for affected areas
upper = np.array([30, 255, 255])  # HSV upper bound
kernel = np.ones((5, 5), np.uint8)

# === STATIC IMAGE ANALYSIS (Leaf disease detection) ===
def analyze_leaf_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Image not found at: {image_path}")
        return

    img = cv2.resize(img, (400, 400))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(img_hsv, lower, upper)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    result = img_rgb.copy()
    result[mask_cleaned > 0] = (255, 0, 0)  # highlight affected in red

    total_pixels = mask_cleaned.size
    affected_pixels = np.count_nonzero(mask_cleaned)
    affected_percent = (affected_pixels / total_pixels) * 100

    # Display results using matplotlib
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original Leaf")
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Initial Segmentation")
    axs[2].imshow(mask_cleaned, cmap='gray')
    axs[2].set_title("Cleaned Mask")
    axs[3].imshow(result)
    axs[3].set_title(f"Result: {affected_percent:.2f}% Affected")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# === REAL-TIME DETECTION USING WEBCAM ===
def run_realtime_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🎥 Real-time detection started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        frame_resized = cv2.resize(frame, (400, 400))
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        result = frame_resized.copy()
        result[cleaned > 0] = (0, 0, 255)  # affected areas in red

        cv2.imshow("Leaf Disease Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === MAIN LOGIC ===
def main():
    print("Select mode:\n1 - Analyze a leaf image\n2 - Real-time webcam detection")
    choice = input("Enter your choice (1 or 2): ")

    if choice.strip() == '1':
        image_path = input("📁 Enter the full path to the leaf image: ").strip()
        analyze_leaf_image(image_path)
    elif choice.strip() == '2':
        run_realtime_detection()
    else:
        print("❌ Invalid choice. Please run the script again and enter 1 or 2.")

# === Run it ===
if __name__ == "__main__":
    main()
