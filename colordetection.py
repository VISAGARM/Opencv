import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Number of partitions for navigation
NUM_PARTITIONS = 7

# Function to determine which partition a point belongs to
def get_partition(x, width):
    for i in range(NUM_PARTITIONS):
        if x < (i + 1) * width // NUM_PARTITIONS:
            return i
    return NUM_PARTITIONS - 1

# Function to detect color
def detect_color(img, color_name):
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    if color_name == "red":
        # Red has two ranges in HSV
        lower_range1 = np.array([0, 100, 100])
        upper_range1 = np.array([10, 255, 255])
        lower_range2 = np.array([170, 100, 100])
        upper_range2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, lower_range1, upper_range1)
        mask2 = cv2.inRange(hsv, lower_range2, upper_range2)
        mask = cv2.bitwise_or(mask1, mask2)
        
    elif color_name == "yellow":
        lower_range = np.array([20, 100, 100])
        upper_range = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        
    elif color_name == "purple":
        lower_range = np.array([125, 50, 50])
        upper_range = np.array([155, 255, 255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # Clean the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    return mask

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Get frame dimensions
    frame_height, frame_width = img.shape[:2]
    frame_center_x = frame_width // 2
    
    # Create display frame
    display_frame = img.copy()
    
    # Draw partition lines
    partition_width = frame_width // NUM_PARTITIONS
    for i in range(1, NUM_PARTITIONS):
        x = i * partition_width
        cv2.line(display_frame, (x, 0), (x, frame_height), (255, 255, 255), 1)
    
    # Draw center reference line
    cv2.line(display_frame, (frame_center_x, 0), (frame_center_x, frame_height), (0, 255, 255), 1)
    
    # Detect colors
    red_mask = detect_color(img, "red")
    yellow_mask = detect_color(img, "yellow")
    purple_mask = detect_color(img, "purple")
    
    # Combine masks for display
    combined_mask = cv2.bitwise_or(red_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, purple_mask)
    
    # Create dictionaries to store color-specific contours
    color_info = {
        "red": {"mask": red_mask, "color_bgr": (0, 0, 255), "detected": False, "info": None},
        "yellow": {"mask": yellow_mask, "color_bgr": (0, 255, 255), "detected": False, "info": None},
        "purple": {"mask": purple_mask, "color_bgr": (128, 0, 128), "detected": False, "info": None}
    }
    
    # Process each color
    for color_name, data in color_info.items():
        contours, _ = cv2.findContours(data["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_area = 0
        largest_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000 and area > largest_area:  # Find largest object > 1000 pixels
                largest_area = area
                largest_contour = contour
        
        if largest_contour is not None:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Get partition
            partition = get_partition(center_x, frame_width)
            
            # Calculate distance (approximation)
            distance = 1000 / w  # Adjust constant as needed
            
            # Calculate offset from center
            offset = center_x - frame_center_x
            
            # Update color info
            data["detected"] = True
            data["info"] = {
                "bbox": (x, y, w, h),
                "center": (center_x, center_y),
                "partition": partition,
                "area": largest_area,
                "distance": distance,
                "offset": offset
            }
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display_frame, (center_x, center_y), 5, data["color_bgr"], -1)
            
            # Add information text
            cv2.putText(display_frame, f"{color_name.upper()}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, data["color_bgr"], 2)
            cv2.putText(display_frame, f"Dist: {distance:.1f}cm", (x, y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, data["color_bgr"], 2)
            cv2.putText(display_frame, f"Part: {partition+1}/{NUM_PARTITIONS}", (x, y - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, data["color_bgr"], 2)
    
    # Display partition labels
    for i in range(NUM_PARTITIONS):
        start_x = i * partition_width
        label_x = start_x + 10
        cv2.putText(display_frame, f"P{i+1}", (label_x, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Generate navigation command based on detected colors
    cmd = "NO TARGET"
    color_targets = []
    
    # Collect all detected color targets
    for color_name, data in color_info.items():
        if data["detected"]:
            color_targets.append((color_name, data["info"]))
    
    # If any target detected, generate command
    if color_targets:
        # Sort by area (largest first)
        color_targets.sort(key=lambda x: x[1]["area"], reverse=True)
        
        # Use the largest detected object
        color_name, target_info = color_targets[0]
        offset = target_info["offset"]
        center_threshold = frame_width // 20  # 5% of frame width
        
        # Determine turn direction and magnitude
        if abs(offset) < center_threshold:
            cmd = f"{color_name.upper()} CENTERED: FORWARD"
        else:
            # Determine if it's a slight or heavy turn
            turn_magnitude = "SLIGHT" if abs(offset) < frame_width // 8 else "HEAVY"
            turn_direction = "RIGHT" if offset < 0 else "LEFT"
            cmd = f"{turn_magnitude} {turn_direction}: {color_name.upper()} TARGET"
    
    # Display command
    cv2.putText(display_frame, f"CMD: {cmd}", (10, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display frames
    cv2.imshow("Navigation Display", display_frame)
    cv2.imshow("Color Masks", combined_mask)
    
    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
