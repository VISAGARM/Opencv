import cv2
import numpy as np
import time
from picamera2 import Picamera2
import serial
import time

# Constants for distance calculation
KNOWN_WIDTH = 10  # Width of the reference object in cm
FOCAL_LENGTH = 500  # Calibrate this for your camera
CENTER_THRESHOLD = 30  # Constants for centering
STOP_DISTANCE = 9  # Distance in cm when robot should stop

# Bluetooth connection to Arduino
bluetooth = serial.Serial("/dev/rfcomm0", 9600)

# Define commands for L293D motor driver
CMD_FORWARD = "1"
CMD_LEFT = "2"
CMD_RIGHT = "3"
CMD_STOP = "4"
CMD_PICK = "5"  # Servo command to pick object

# Track the current movement state
current_movement = CMD_STOP  # Start with stopped state

# Define detectable objects as a list of dictionaries
# This makes it easy to add more objects later
DETECTABLE_OBJECTS = [
    {
        'name': 'Tape',
        'hsv_lower': np.array([0, 83, 78]),
        'hsv_upper': np.array([53, 215, 255]),
        'color': (0, 255, 255)  # Yellow color for display (BGR)
    },
    {
        'name': 'Tissue',
        'hsv_lower': np.array([0, 0, 228]),
        'hsv_upper': np.array([137, 148, 255]),
        'color': (255, 0, 255)  # Magenta color for display (BGR)
    }
    # Add more objects easily by copying this template:
    # {
    #     'name': 'Object_Name',
    #     'hsv_lower': np.array([h_min, s_min, v_min]),
    #     'hsv_upper': np.array([h_max, s_max, v_max]),
    #     'color': (B, G, R)
    # }
]

def calculate_distance(pixel_width):
    if pixel_width == 0:
        return float('inf')
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

def get_turning_guidance(center_x, frame_center_x):
    diff = center_x - frame_center_x
    if abs(diff) <= CENTER_THRESHOLD:
        return "Centered", 0
    turning_amount = diff / 10  # Approx. 1° per 10 pixels
    return ("Turn RIGHT", turning_amount) if diff > 0 else ("Turn LEFT", abs(turning_amount))

def send_command(cmd, force_stop=True):
    """
    Send command to Arduino through Bluetooth
    If force_stop is True, always send stop command first before changing direction
    """
    global current_movement
    
    try:
        # If we're changing direction and force_stop is True, send stop command first
        if force_stop and cmd != CMD_STOP and current_movement != CMD_STOP:
            bluetooth.write(CMD_STOP.encode())
            print(f"Command sent: {CMD_STOP} (stopping before direction change)")
            time.sleep(0.2)  # Short delay to ensure motor stops
        
        # Send the actual command
        bluetooth.write(cmd.encode())
        print(f"Command sent: {cmd}")
        
        # Update current movement state
        current_movement = cmd
        return True
    except Exception as e:
        print(f"Failed to send command: {e}")
        return False

def detect_objects_from_masks(hsv_frame):
    """
    Detect objects from HSV masks based on the DETECTABLE_OBJECTS list
    Returns a combined mask of all objects and individual object masks
    """
    combined_mask = None
    object_masks = []
    
    for obj in DETECTABLE_OBJECTS:
        # Create mask for this object
        obj_mask = cv2.inRange(hsv_frame, obj['hsv_lower'], obj['hsv_upper'])
        object_masks.append(obj_mask)
        
        # Add to combined mask
        if combined_mask is None:
            combined_mask = obj_mask.copy()
        else:
            combined_mask = cv2.bitwise_or(combined_mask, obj_mask)
    
    return combined_mask, object_masks

def detect_objects_realtime():
    """Real-time detection of objects with L293D robot control."""
    
    # Initialize PiCamera2
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    time.sleep(1.0)  # Give camera time to warm up

    frame_width, frame_height = 640, 480
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    print(f"PiCamera2 resolution: {frame_width}x{frame_height}")
    print("Press 'q' to quit")

    prev_time = time.time()
    avg_fps = 0
    fps_count = 0
    
    # Area threshold for picking object
    PICK_AREA_THRESHOLD = 5000  # Adjust based on your requirements
    
    # Robot state
    robot_state = "SEARCHING"  # States: SEARCHING, STOPPING, PICKING
    last_command_time = 0
    command_cooldown = 0.5  # Seconds between commands
    picking_start_time = 0
    
    # Flag to track if servo is returning to initial position after picking
    servo_returning = False
    servo_return_start_time = 0
    SERVO_RETURN_TIME = 5.0  # Time in seconds for servo to return to initial position

    def draw_center_zone(frame):
        cv2.rectangle(
            frame,
            (frame_center_x - CENTER_THRESHOLD, frame_center_y - CENTER_THRESHOLD),
            (frame_center_x + CENTER_THRESHOLD, frame_center_y + CENTER_THRESHOLD),
            (0, 255, 0), 1
        )

    # For running average of detected objects (smoothing)
    prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
    smoothing_factor = 0.7  # Higher values give more weight to previous positions

    while True:
        try:
            # Capture frame
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                print("Failed to capture frame")
                continue
                
            # Convert RGB to BGR (OpenCV standard)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # Calculate FPS with smoothing
            current_time = time.time()
            if prev_time > 0:
                instantaneous_fps = 1 / (current_time - prev_time)
                if fps_count < 10:
                    avg_fps = (avg_fps * fps_count + instantaneous_fps) / (fps_count + 1)
                    fps_count += 1
                else:
                    avg_fps = 0.9 * avg_fps + 0.1 * instantaneous_fps
            prev_time = current_time

            # Convert to HSV color space
            hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Get object masks using our detection function
            combined_mask, object_masks = detect_objects_from_masks(hsv_frame)

            # Enhanced morphological operations for noise reduction
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
            
            # Debug: Show mask
            cv2.imshow("Mask", combined_mask)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw crosshair and center zone
            cv2.line(frame, (frame_center_x, frame_center_y - 20), (frame_center_x, frame_center_y + 20), (255, 255, 255), 2)
            cv2.line(frame, (frame_center_x - 20, frame_center_y), (frame_center_x + 20, frame_center_y), (255, 255, 255), 2)
            draw_center_zone(frame)

            # Process contours
            contour_found = False
            candidate_contours = []
            min_area_threshold = 300  # Minimum area threshold
            
            # Collect all valid contours with their properties
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Determine object type by checking which mask has the most overlap
                    contour_mask = np.zeros_like(combined_mask)
                    cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                    
                    max_overlap = 0
                    detected_obj_index = 0
                    
                    # Check each object mask for overlap
                    for i, obj_mask in enumerate(object_masks):
                        overlap = cv2.bitwise_and(contour_mask, obj_mask)
                        overlap_amount = cv2.countNonZero(overlap)
                        
                        if overlap_amount > max_overlap:
                            max_overlap = overlap_amount
                            detected_obj_index = i
                    
                    # Get object info
                    obj_type = DETECTABLE_OBJECTS[detected_obj_index]['name']
                    obj_color = DETECTABLE_OBJECTS[detected_obj_index]['color']
                    
                    # Estimate distance based on width
                    estimated_distance = calculate_distance(w)
                    
                    # Store contour with its properties
                    candidate_contours.append({
                        'contour': contour,
                        'area': area,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'distance': estimated_distance,
                        'type': obj_type,
                        'color': obj_color
                    })
            
            # Sort contours by distance (closest first)
            candidate_contours.sort(key=lambda c: c['distance'])
            
            # State machine for robot control
            now = time.time()
            
            # Handle the servo returning state
            if servo_returning:
                if now - servo_return_start_time > SERVO_RETURN_TIME:
                    servo_returning = False
                    robot_state = "SEARCHING"
                    print("Servo returned to initial position, resuming search")
                else:
                    # Stay stopped while servo is returning
                    send_command(CMD_STOP, force_stop=False)
            
            # Handle state transitions and commands
            elif robot_state == "PICKING":
                # Check if picking operation is complete
                if now - picking_start_time > 5:  # 5 seconds for picking operation
                    # Change to servo returning state
                    servo_returning = True
                    servo_return_start_time = now
                    print("Pick complete, waiting for servo to return")
            
            elif robot_state == "STOPPING":
                # After stopping, initiate picking if an object is close enough
                if now - last_command_time > 1.0:  # Wait for robot to fully stop
                    if candidate_contours and candidate_contours[0]['distance'] <= STOP_DISTANCE:
                        send_command(CMD_PICK, force_stop=False)  # No need to stop before picking
                        robot_state = "PICKING"
                        picking_start_time = now
                        print("Starting pick operation")
                    else:
                        robot_state = "SEARCHING"
                        print("No object in range after stopping, resuming search")
            
            elif robot_state == "SEARCHING":
                # Process the closest contour if any were found
                if candidate_contours and now - last_command_time > command_cooldown:
                    contour_found = True
                    closest_contour = candidate_contours[0]
                    
                    # Get properties of the closest contour
                    x, y, w, h = closest_contour['x'], closest_contour['y'], closest_contour['w'], closest_contour['h']
                    area = closest_contour['area']
                    obj_type = closest_contour['type']
                    obj_color = closest_contour['color']
                    
                    # Apply smoothing to reduce jitter
                    if prev_w > 0:  # If we have previous measurements
                        x = int(smoothing_factor * prev_x + (1 - smoothing_factor) * x)
                        y = int(smoothing_factor * prev_y + (1 - smoothing_factor) * y)
                        w = int(smoothing_factor * prev_w + (1 - smoothing_factor) * w)
                        h = int(smoothing_factor * prev_h + (1 - smoothing_factor) * h)
                    
                    # Update previous values
                    prev_x, prev_y, prev_w, prev_h = x, y, w, h
                    
                    # Calculate center of contour
                    center_x = x + w // 2
                    
                    # Calculate distance using apparent width
                    distance = closest_contour['distance']
                    
                    # Get turning guidance
                    turn_direction, turn_amount = get_turning_guidance(center_x, frame_center_x)
                    
                    # Object detected within stopping distance and centered
                    if distance <= STOP_DISTANCE and turn_direction == "Centered" and area > PICK_AREA_THRESHOLD:
                        # Object is close enough, stop the robot
                        send_command(CMD_STOP, force_stop=False)  # No need to stop before stopping
                        robot_state = "STOPPING"
                        last_command_time = now
                        print(f"Object detected within {distance:.1f}cm - stopping")
                    else:
                        # Navigation commands based on position
                        if turn_direction == "Centered":
                            # If we're already going forward, no need to stop first
                            if current_movement == CMD_FORWARD:
                                force_stop = False
                            else:
                                force_stop = True
                            send_command(CMD_FORWARD, force_stop)
                            print(f"Moving forward - object at {distance:.1f}cm")
                        elif turn_direction == "Turn LEFT":
                            # If we're changing direction, stop first
                            send_command(CMD_LEFT, force_stop=True)
                            print(f"Turning left - {turn_amount:.1f}°")
                        elif turn_direction == "Turn RIGHT":
                            # If we're changing direction, stop first
                            send_command(CMD_RIGHT, force_stop=True)
                            print(f"Turning right - {turn_amount:.1f}°")
                        last_command_time = now
                elif not candidate_contours and now - last_command_time > command_cooldown:
                    # No objects found, stop the robot if not already stopped
                    if current_movement != CMD_STOP:
                        send_command(CMD_STOP, force_stop=False)  # No need to stop before stopping
                        last_command_time = now
                        print("No objects detected - stopping")

            # Visualization for all contours
            for i, contour_data in enumerate(candidate_contours):
                x, y, w, h = contour_data['x'], contour_data['y'], contour_data['w'], contour_data['h']
                area = contour_data['area']
                distance = contour_data['distance']
                obj_type = contour_data['type']
                obj_color = contour_data['color']
                
                # Color based on if it's the closest (brighten) or not
                if i == 0:  # Closest object
                    color = obj_color
                else:
                    # Make color dimmer for non-closest objects
                    color = tuple(int(c/2) for c in obj_color)
                
                # Draw rectangle around object
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Only add detailed info for closest object
                if i == 0:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Display distance and area information
                    dist_text = f"Distance: {distance:.1f}cm"
                    cv2.putText(frame, dist_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, dist_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Display area information
                    area_text = f"Area: {area:.1f} px²"
                    cv2.putText(frame, area_text, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, area_text, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Display type information
                    type_text = f"Type: {obj_type}"
                    cv2.putText(frame, type_text, (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, type_text, (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Get turning guidance
                    turn_direction, turn_amount = get_turning_guidance(center_x, frame_center_x)
                    
                    # Display turning guidance or centered status
                    if turn_direction == "Centered":
                        guidance_text = f"Centered - {distance:.1f}cm"
                        text_color = (0, 255, 0)  # Green for centered
                    else:
                        guidance_text = f"{turn_direction} {turn_amount:.1f}°"
                        text_color = (0, 255, 255)  # Yellow for turning
                    
                    # Draw with outline for better visibility
                    cv2.putText(frame, guidance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, guidance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                    
                    # Draw line from center to object
                    cv2.line(frame, (frame_center_x, frame_center_y), (center_x, center_y), color, 2)
                else:
                    # Minimal info for other objects
                    cv2.putText(frame, f"#{i+1}: {obj_type}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Reset smoothing values if no object detected
            if not candidate_contours:
                prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
            
            # Display message if no objects detected
            if not contour_found:
                cv2.putText(frame, "No object detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display current robot state
            state_display = robot_state
            if servo_returning:
                state_display = "SERVO RETURNING"
                
            cv2.putText(frame, f"State: {state_display}", (frame_width - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                       
            # Display current movement command
            cv2.putText(frame, f"Movement: {current_movement}", (frame_width - 200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display detected objects list
            cv2.putText(frame, "Detecting:", (10, frame_height - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            for i, obj in enumerate(DETECTABLE_OBJECTS):
                cv2.putText(frame, f"- {obj['name']}", (30, frame_height - 50 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj['color'], 2)

            # Show main detection window
            cv2.imshow("Detection", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Make sure to stop the robot when exiting
                send_command(CMD_STOP, force_stop=False)
                break
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            # Safety: stop robot on error
            send_command(CMD_STOP, force_stop=False)

    # Clean up
    cv2.destroyAllWindows()
    picam2.stop()
    bluetooth.close()

# Run the main program
if __name__ == "__main__":
    detect_objects_realtime()
