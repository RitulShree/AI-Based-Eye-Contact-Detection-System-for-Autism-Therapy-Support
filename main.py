import cv2 as cv
import mediapipe as mp
import numpy as np
import csv
import os
import time

from behavior_analyzer import BehaviorAnalyzer

LEFT_IRIS = [474, 475, 476, 477]
LEFT_EYE_LEFT_CORNER = 33
LEFT_EYE_RIGHT_CORNER = 133
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_EYE_LEFT_CORNER = 362
RIGHT_EYE_RIGHT_CORNER = 263

# -------- Face Mesh --------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --------EAR---------
def calculate_ear(eye_points):
    eye = np.array(eye_points, dtype=np.float32)

    # vertical distances
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])

    # horizontal distance
    h = np.linalg.norm(eye[0] - eye[3])

    ear = (v1 + v2) / (2.0 * h)
    return ear


LEFT_EYE = [33, 159, 158, 133, 153, 145]
RIGHT_EYE = [362, 386, 387, 263, 373, 374]

def get_landmark_coords(landmarks, index, w, h):
    lm = landmarks[index]
    return int(lm.x * w), int(lm.y * h)

def get_iris_center(landmarks, iris_indices, w, h):
    points = [get_landmark_coords(landmarks, i, w, h) for i in iris_indices]
    x = int(np.mean([p[0] for p in points]))
    y = int(np.mean([p[1] for p in points]))
    return x, y

def get_gaze_ratio(iris_x, left_x, right_x):
    eye_width = right_x - left_x
    if eye_width == 0:
        return 0.5
    return (iris_x - left_x) / eye_width

# -------- Video --------

cap = cv.VideoCapture(0)
fps = cap.get(cv.CAP_PROP_FPS)

# Some webcams return 0, so safeguard:
if fps == 0:
    fps = 30
analyzer = BehaviorAnalyzer(fps=fps)

def save_session_to_csv(data, file_path="dataset/gaze_dataset.csv"):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "session_id",
                "session_type",
                "duration",
                "blink_rate",
                "avg_closure_duration",
                "avg_IBI",
                "IBI_std",
                "long_closures",
                "total_fixations",
                "avg_fixation_duration",
                "avg_movement",
                "gaze_variance",
                "longest_no_blink",
                "eye_contact_time",
                "eye_contact_percentage"
            ])

        writer.writerow(data)

def rescale(img, scale=0.5):
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    return cv.resize(img, (w, h), interpolation=cv.INTER_AREA)






# -------- Layer 2 Variables --------
centroid_history = []
movement_history = []

WINDOW_SECONDS = 2
SESSION_DURATION = 60  # seconds

gaze_history = []
analyzer.eye_contact_frames = 0
analyzer.total_frames = 0
session_done = False

def get_head_pose(landmarks, w, h):
    # Key points
    nose = get_landmark_coords(landmarks, 1, w, h)
    chin = get_landmark_coords(landmarks, 152, w, h)
    left_eye = get_landmark_coords(landmarks, 33, w, h)
    right_eye = get_landmark_coords(landmarks, 263, w, h)

    # Convert to numpy
    nose = np.array(nose)
    chin = np.array(chin)
    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)

    # Horizontal head direction
    eye_center = (left_eye + right_eye) / 2
    dx = nose[0] - eye_center[0]

    # Vertical direction
    dy = nose[1] - eye_center[1]

    return dx, dy

while True:
    ret, img = cap.read()
    

   
    if not ret:
         print("Video ended or cannot read frame")
         break
    analyzer.total_frames += 1

    # img = rescale(img)
    height, width, _ = img.shape

    # -------- Convert to RGB --------
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # -------- Process Face --------
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
     for face_landmarks in results.multi_face_landmarks:

        left_eye_pts = []
        right_eye_pts = []

        for idx in LEFT_EYE:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * width)
            y = int(lm.y * height)
            left_eye_pts.append((x, y))
            cv.circle(img, (x, y), 2, (0, 255, 0), -1)

        for idx in RIGHT_EYE:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * width)
            y = int(lm.y * height)
            right_eye_pts.append((x, y))
            cv.circle(img, (x, y), 2, (0, 255, 0), -1)

        ear_left = calculate_ear(left_eye_pts)
        ear_right = calculate_ear(right_eye_pts)

        ear_avg = (ear_left + ear_right) / 2
        analyzer.update_blink(ear_avg)

        # ---- Eye Centroid ----
        all_eye_points = left_eye_pts + right_eye_pts
        xs = [pt[0] for pt in all_eye_points]
        ys = [pt[1] for pt in all_eye_points]

        centroid_x = int(np.mean(xs))
        centroid_y = int(np.mean(ys))

        cv.circle(img, (centroid_x, centroid_y), 4, (0, 0, 255), -1)

        centroid_history.append((centroid_x, centroid_y))
        # Keep only last 1 second of centroid history
        if len(centroid_history) > fps:
            centroid_history.pop(0)


        # ---- Movement Calculation ----
        if len(centroid_history) >= 2:
             prev_x, prev_y = centroid_history[-2]
             curr_x, curr_y = centroid_history[-1]

             movement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
             analyzer.update_gaze(movement)
             movement_history.append(movement)

             


        # ---- Gaze Stability (Sliding Window) ----
        window_size = int(fps * WINDOW_SECONDS)

        if len(movement_history) >= window_size:
             recent_movements = movement_history[-window_size:]
             gaze_stability_score = np.var(recent_movements)
        else:
             gaze_stability_score = 0         

        # ---- IRIS BASED GAZE (IMPROVED: BOTH EYES) ----

        # LEFT EYE
        left_corner_L = get_landmark_coords(face_landmarks.landmark, LEFT_EYE_LEFT_CORNER, width, height)
        right_corner_L = get_landmark_coords(face_landmarks.landmark, LEFT_EYE_RIGHT_CORNER, width, height)
        iris_L = get_iris_center(face_landmarks.landmark, LEFT_IRIS, width, height)

        # RIGHT EYE
        left_corner_R = get_landmark_coords(face_landmarks.landmark, RIGHT_EYE_LEFT_CORNER, width, height)
        right_corner_R = get_landmark_coords(face_landmarks.landmark, RIGHT_EYE_RIGHT_CORNER, width, height)
        iris_R = get_iris_center(face_landmarks.landmark, RIGHT_IRIS, width, height)

        # Draw iris points
        cv.circle(img, iris_L, 3, (0, 255, 0), -1)
        cv.circle(img, iris_R, 3, (0, 255, 0), -1)

        # Compute gaze ratios
        ratio_L = get_gaze_ratio(iris_L[0], left_corner_L[0], right_corner_L[0])
        ratio_R = get_gaze_ratio(iris_R[0], left_corner_R[0], right_corner_R[0])

        # Average both eyes
        gaze_ratio = (ratio_L + ratio_R) / 2

        gaze_history.append(gaze_ratio)

        if len(gaze_history) > 5:
            gaze_history.pop(0)

        gaze_ratio = np.mean(gaze_history)

        if gaze_ratio < 0.30:
            gaze_direction = "LEFT"
        elif gaze_ratio > 0.70:
            gaze_direction = "RIGHT"
        else:
            gaze_direction = "CENTER"
        
        
        
        

        
        dx, dy = get_head_pose(face_landmarks.landmark, width, height)

        face_width = abs(right_eye_pts[0][0] - left_eye_pts[0][0])

        if face_width != 0:
            normalized_dx = dx / face_width
        else:
            normalized_dx = 0

        if abs(normalized_dx) < 0.1:
            head_direction = "CENTER"
        elif normalized_dx > 0:
            head_direction = "RIGHT"
        else:
            head_direction = "LEFT"

        # ---- TRUE EYE CONTACT ----
        if gaze_direction == "CENTER" and head_direction == "CENTER":
            analyzer.eye_contact_frames += 1
        
        # Eye center
        eye_center = ((left_eye_pts[0][0] + right_eye_pts[0][0]) // 2,
                      (left_eye_pts[0][1] + right_eye_pts[0][1]) // 2)
        face_width = abs(right_eye_pts[0][0] - left_eye_pts[0][0])
        # Use dx to draw direction
        scale = face_width * 1  # you can tweak this

        # Clamp dx to avoid huge lines
        dx_clamped = max(min(dx, 1), -1)

        end_point = (int(eye_center[0] + dx_clamped * scale),
             int(eye_center[1]))

        cv.line(img, eye_center, end_point, (255, 0, 255), 2)
       
       
        cv.putText(
            img,
            f"EAR: {ear_avg:.2f}",
            (30, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        current_time = time.time() - analyzer.start_time
        

        # ---- Auto Stop Session After Fixed Time ----
        if current_time >= SESSION_DURATION:
           print("Session duration reached. Ending session...")
           session_done = True
           break

        if current_time > 0:
             blink_rate_live = (analyzer.blink_count / current_time) * 60
        else:
             blink_rate_live = 0
        cv.putText(img, f"Blinks: {analyzer.blink_count}", (30, 60),
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.putText(img, f"Blink Rate: {blink_rate_live:.2f}/min", (30, 90),
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(analyzer.closure_durations) > 0:
           last_duration = analyzer.closure_durations[-1]
        else:
           last_duration = 0

        cv.putText(img, f"Last Closure: {last_duration:.3f}s",
           (30, 120),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (0, 255, 0),
           2)
        
        if len(analyzer.ibi_values) > 0:
          last_ibi = analyzer.ibi_values[-1]
        else:
           last_ibi = 0

        cv.putText(img, f"IBI: {last_ibi:.2f}s",
           (30, 150),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (0, 255, 0),
           2)
       # ---- Longest No-Blink Interval ----
        if len(analyzer.ibi_values) > 0:
           longest_no_blink = max(analyzer.ibi_values)
        else:
           longest_no_blink = 0

        cv.putText(img, f"Longest No Blink: {longest_no_blink:.2f}s",
           (30, 180),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (0, 255, 0),
           2)


         # ---- Blink Variability (IBI Std Dev) ----
        if len(analyzer.ibi_values) > 1:
          blink_variability = np.std(analyzer.ibi_values)
        else:
          blink_variability = 0

        cv.putText(img, f"Blink Variability: {blink_variability:.2f}",
           (30, 210),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (0, 255, 0),
           2)
        cv.putText(img, f"Gaze Stability: {gaze_stability_score:.2f}",
           (30, 240),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (0, 255, 255),
           2)
        

        cv.putText(img, f"Fixations: {analyzer.fixation_count}",
           (30, 270),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (255, 255, 0),
           2)

        if len(analyzer.fixation_durations) > 0:
             avg_fixation_duration = np.mean(analyzer.fixation_durations)
        else:
             avg_fixation_duration = 0

        cv.putText(img, f"Avg Fix Dur: {avg_fixation_duration:.2f}s",
           (30, 300),
           cv.FONT_HERSHEY_SIMPLEX,
           0.7,
           (255, 255, 0),
           2)
        eye_contact_pct = (analyzer.eye_contact_frames / analyzer.total_frames) * 100 if analyzer.total_frames > 0 else 0

        cv.putText(img,
          f"Eye Contact (REAL): {eye_contact_pct:.1f}%",
          (30, 330),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2)

        cv.putText(img,
            f"Gaze: {gaze_direction}",
            (30, 360),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2)
        cv.putText(img, f"Ratio: {gaze_ratio:.2f}", (30, 390),
           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv.putText(img,
            f"Head: {head_direction}",
            (30, 420),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2)
    
    if session_done:
     break
    
    # -------- Show Frame --------
    cv.imshow("Face Mesh Video", img)

    # ESC to quit
    if cv.waitKey(1) & 0xFF == 27:
        break
        
# -------- Session Summary --------

metrics = analyzer.get_session_metrics()

print("\n----- SESSION SUMMARY -----")
print(f"Session Duration: {metrics['duration']:.2f} seconds")
print(f"Average Blink Rate: {metrics['blink_rate']:.2f} per minute")
print(f"Average Closure Duration: {metrics['avg_closure_duration']:.3f} seconds")
print(f"Average IBI: {metrics['avg_IBI']:.3f} seconds")
print(f"Longest No-Blink Interval: {metrics['longest_no_blink']:.3f} seconds")
print(f"Blink Variability (IBI Std Dev): {metrics['IBI_std']:.3f}")
print(f"Long Closures Count (>0.5s): {metrics['long_closures']}")
print(f"Total Fixations: {metrics['total_fixations']}")
print(f"Average Fixation Duration: {metrics['avg_fixation_duration']:.3f} seconds")
print(f"Average Eye Movement: {metrics['avg_movement']:.3f} pixels")
print(f"Gaze Stability Variance: {metrics['gaze_variance']:.3f}")
print("----------------------------\n")

# -------- Ask Session Type --------
while True:
    session_type = input("Enter session type (typical / atypical): ").strip().lower()
    if session_type in ["typical", "atypical"]:
        break
    else:
        print("Invalid input. Please enter typical or atypical.")
# -------- Save To CSV --------
session_id = int(time.time())

data_row = [
    session_id,
    session_type,          # ← ADD THIS (VERY IMPORTANT)
   metrics['duration'],
metrics['blink_rate'],
metrics['avg_closure_duration'],
metrics['avg_IBI'],
metrics['IBI_std'],
metrics['long_closures'],
metrics['total_fixations'],
metrics['avg_fixation_duration'],
metrics['avg_movement'],
metrics['gaze_variance'],
metrics['longest_no_blink'],
metrics['eye_contact_time'],
metrics['eye_contact_percentage']
]

save_session_to_csv(data_row)
print("FULL PATH:", os.path.abspath("dataset/gaze_dataset.csv"))
print("FILE EXISTS:", os.path.isfile("dataset/gaze_dataset.csv"))

print("Session saved to CSV successfully.")

cap.release()
cv.destroyAllWindows()
