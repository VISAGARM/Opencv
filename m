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

def send_command(cmd):
    """Send command to Arduino through Bluetooth"""
    try:
        bluetooth.write(cmd.encode())
        print(f"Command sent: {cmd}")
        return True
    except Exception as e:
        print(f"Failed to send command: {e}")
        return False

def detect_objects_realtime():
    """Real-time detection of tape and tissue objects with L293D robot control."""
    
    # HSV values for tape and tissue
    TAPE_HSV_LOWER = np.array([0, 83, 78])
    TAPE_HSV_UPPER = np.array([53, 215, 255])
    
    TISSUE_HSV_LOWER = np.array([0, 0, 228])
    TISSUE_HSV_UPPER = np.array([137, 148, 255])
    
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

            # Create masks for tape and tissue
            tape_mask = cv2.inRange(hsv_frame, TAPE_HSV_LOWER, TAPE_HSV_UPPER)
            tissue_mask = cv2.inRange(hsv_frame, TISSUE_HSV_LOWER, TISSUE_HSV_UPPER)
            
            # Combine masks to detect both objects
            combined_mask = cv2.bitwise_or(tape_mask, tissue_mask)

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
                    
                    # Determine object type (tape or tissue)
                    contour_mask = np.zeros_like(combined_mask)
                    cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                    
                    tape_overlap = cv2.bitwise_and(contour_mask, tape_mask)
                    tissue_overlap = cv2.bitwise_and(contour_mask, tissue_mask)
                    
                    tape_pixels = cv2.countNonZero(tape_overlap)
                    tissue_pixels = cv2.countNonZero(tissue_overlap)
                    
                    obj_type = "Tape" if tape_pixels > tissue_pixels else "Tissue"
                    
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
                        'type': obj_type
                    })
            
            # Sort contours by distance (closest first)
            candidate_contours.sort(key=lambda c: c['distance'])
            
            # State machine for robot control
            now = time.time()
            
            # Handle state transitions and commands
            if robot_state == "PICKING":
                # Check if picking operation is complete
                if now - picking_start_time > 5:  # 5 seconds for picking operation
                    robot_state = "SEARCHING"
                    print("Picking complete, resuming search")
            
            elif robot_state == "STOPPING":
                # After stopping, initiate picking if an object is close enough
                if now - last_command_time > 1.0:  # Wait for robot to fully stop
                    if candidate_contours and candidate_contours[0]['distance'] <= STOP_DISTANCE:
                        send_command(CMD_PICK)
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
                        send_command(CMD_STOP)
                        robot_state = "STOPPING"
                        last_command_time = now
                        print(f"Object detected within {distance:.1f}cm - stopping")
                    else:
                        # Navigation commands based on position
                        if turn_direction == "Centered":
                            send_command(CMD_FORWARD)
                            print(f"Moving forward - object at {distance:.1f}cm")
                        elif turn_direction == "Turn LEFT":
                            send_command(CMD_LEFT)
                            print(f"Turning left - {turn_amount:.1f}°")
                        elif turn_direction == "Turn RIGHT":
                            send_command(CMD_RIGHT)
                            print(f"Turning right - {turn_amount:.1f}°")
                        last_command_time = now
                elif not candidate_contours and now - last_command_time > command_cooldown:
                    # No objects found, stop the robot
                    send_command(CMD_STOP)
                    last_command_time = now
                    print("No objects detected - stopping")

            # Visualization for all contours
            for i, contour_data in enumerate(candidate_contours):
                x, y, w, h = contour_data['x'], contour_data['y'], contour_data['w'], contour_data['h']
                area = contour_data['area']
                distance = contour_data['distance']
                obj_type = contour_data['type']
                
                # Color based on object type and if it's the closest
                if i == 0:  # Closest object
                    color = (0, 255, 255) if obj_type == "Tape" else (255, 0, 255)  # Yellow for tape, magenta for tissue
                else:
                    color = (0, 128, 128) if obj_type == "Tape" else (128, 0, 128)  # Dimmer colors for other objects
                
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
            cv2.putText(frame, f"State: {robot_state}", (frame_width - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show main detection window
            cv2.imshow("Detection", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Make sure to stop the robot when exiting
                send_command(CMD_STOP)
                break
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            # Safety: stop robot on error
            send_command(CMD_STOP)

    # Clean up
    cv2.destroyAllWindows()
    picam2.stop()
    bluetooth.close()

# Arduino code for L293D motor driver and servo control
def generate_arduino_code():
    """Generate Arduino code for L293D motor control and servo pick sequence"""
    arduino_code = """
    #include <Servo.h>
    #include <SoftwareSerial.h>
    
    // Bluetooth module connection
    SoftwareSerial BTSerial(10, 11); // RX, TX
    
    // L293D motor driver pins
    #define ENA 5 // Enable motor A
    #define ENB 6 // Enable motor B
    #define IN1 2 // Motor A input 1
    #define IN2 3 // Motor A input 2
    #define IN3 4 // Motor B input 1
    #define IN4 7 // Motor B input 2
    
    // Define servo pins
    #define SERVO1_PIN 8
    #define SERVO2_PIN 9
    #define SERVO3_PIN 12
    #define SERVO4_PIN 13
    #define SERVO5_PIN A0
    
    // Motor speed
    #define MOTOR_SPEED 200 // 0-255
    #define TURN_SPEED 150 // Slower for turns
    
    // Create servo objects
    Servo servo1;
    Servo servo2;
    Servo servo3;
    Servo servo4;
    Servo servo5;
    
    // Initial positions
    int initialPos1 = 90;
    int initialPos2 = 90;
    int initialPos3 = 90;
    int initialPos4 = 90;
    int initialPos5 = 90;
    
    void setup() {
      Serial.begin(9600);
      BTSerial.begin(9600);
      
      // Setup L293D motor driver pins
      pinMode(ENA, OUTPUT);
      pinMode(ENB, OUTPUT);
      pinMode(IN1, OUTPUT);
      pinMode(IN2, OUTPUT);
      pinMode(IN3, OUTPUT);
      pinMode(IN4, OUTPUT);
      
      // Stop motors at startup
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
      
      // Attach servos
      servo1.attach(SERVO1_PIN);
      servo2.attach(SERVO2_PIN);
      servo3.attach(SERVO3_PIN);
      servo4.attach(SERVO4_PIN);
      servo5.attach(SERVO5_PIN);
      
      // Move servos to initial positions
      resetServos();
      
      Serial.println("Robot ready!");
    }
    
    void loop() {
      if (BTSerial.available()) {
        char cmd = BTSerial.read();
        executeCommand(cmd);
      }
    }
    
    void executeCommand(char cmd) {
      Serial.print("Command received: ");
      Serial.println(cmd);
      
      switch(cmd) {
        case '1': // Forward
          moveForward();
          break;
          
        case '2': // Left
          turnLeft();
          break;
          
        case '3': // Right
          turnRight();
          break;
          
        case '4': // Stop
          stopMotors();
          break;
          
        case '5': // Pick with servos
          pickObject();
          break;
          
        default:
          Serial.println("Unknown command");
          break;
      }
    }
    
    void moveForward() {
      Serial.println("Moving forward");
      analogWrite(ENA, MOTOR_SPEED);
      analogWrite(ENB, MOTOR_SPEED);
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
    }
    
    void turnLeft() {
      Serial.println("Turning left");
      analogWrite(ENA, TURN_SPEED);
      analogWrite(ENB, TURN_SPEED);
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
    }
    
    void turnRight() {
      Serial.println("Turning right");
      analogWrite(ENA, TURN_SPEED);
      analogWrite(ENB, TURN_SPEED);
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
    }
    
    void stopMotors() {
      Serial.println("Stopping motors");
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, LOW);
    }
    
    void resetServos() {
      servo1.write(initialPos1);
      servo2.write(initialPos2);
      servo3.write(initialPos3);
      servo4.write(initialPos4);
      servo5.write(initialPos5);
      delay(500);
    }
    
    void pickObject() {
      Serial.println("Picking object");
      
      // Make sure motors are stopped before picking
      stopMotors();
      delay(500);
      
      // Sequence for picking object with 5 servos
      // Step 1: Open gripper (Servo 1)
      servo1.write(160);
      delay(500);
      
      // Step 2: Position arm (Servo 2 & 3)
      servo2.write(45);
      delay(300);
      servo3.write(135);
      delay(500);
      
      // Step 3: Lower arm (Servo 4 & 5)
      servo4.write(45);
      delay(300);
      servo5.write(135);
      delay(800);
      
      // Step 4: Close gripper (Servo 1)
      servo1.write(90);
      delay(1000);
      
      // Step 5: Lift arm (Servo 4 & 5)
      servo4.write(90);
      delay(300);
      servo5.write(90);
      delay(500);
      
      // Step 6: Return arm to initial position (Servo 2 & 3)
      servo2.write(90);
      delay(300);
      servo3.write(90);
      delay(500);
      
      Serial.println("Pick complete");
    }
    """
    return arduino_code

# Run the main program
if __name__ == "__main__":
    detect_objects_realtime()
