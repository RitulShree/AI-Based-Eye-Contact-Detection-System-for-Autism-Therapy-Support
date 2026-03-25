import numpy as np
import time

class BehaviorAnalyzer:

    def __init__(self, fps=30):
        self.fps = fps
        self.total_frames = 0
        self.eye_contact_frames = 0
        self.EYE_CONTACT_THRESH = 0.27
        # ---- Blink detection parameters ----
        self.EYE_AR_THRESH = 0.25
        self.CONSEC_FRAMES = 3

        # ---- Blink state ----
        self.blink_frame_counter = 0
        self.blink_count = 0

        self.closure_durations = []
        self.long_closure_count = 0
        self.long_closure_durations = []

        self.blink_timestamps = []
        self.ibi_values = []

        # ---- Fixations ----
        self.fixation_frame_counter = 0
        self.FIXATION_THRESHOLD = 2
        self.FIXATION_MIN_DURATION = 0.3
        self.fixation_count = 0
        self.fixation_durations = []

        # ---- Gaze movement ----
        self.movement_history = []

        # ---- Timing ----
        self.start_time = time.time()

    # ---------------- BLINK UPDATE ----------------
    def update_blink(self, ear_avg):

        


        if ear_avg < self.EYE_AR_THRESH:
            self.blink_frame_counter += 1

        else:

            if self.blink_frame_counter >= self.CONSEC_FRAMES:

                closure_time = self.blink_frame_counter / self.fps

                # ---- Spontaneous blink ----
                if 0.08 <= closure_time <= 0.5:

                    self.blink_count += 1
                    self.closure_durations.append(closure_time)

                    current_time = time.time() - self.start_time
                    self.blink_timestamps.append(current_time)

                    if len(self.blink_timestamps) >= 2:
                        ibi = self.blink_timestamps[-1] - self.blink_timestamps[-2]
                        self.ibi_values.append(ibi)

                # ---- Long closure ----
                elif closure_time > 0.5:

                    self.long_closure_count += 1
                    self.long_closure_durations.append(closure_time)

            self.blink_frame_counter = 0

    # ---------------- GAZE UPDATE ----------------
    def update_gaze(self, movement_value):

        self.movement_history.append(movement_value)

        if movement_value < self.FIXATION_THRESHOLD:
            self.fixation_frame_counter += 1
        else:

            if self.fixation_frame_counter > 0:

                fixation_duration = self.fixation_frame_counter / self.fps

                if fixation_duration >= self.FIXATION_MIN_DURATION:
                    self.fixation_count += 1
                    self.fixation_durations.append(fixation_duration)

            self.fixation_frame_counter = 0

    # ---------------- METRICS ----------------
    def get_session_metrics(self):
        eye_contact_percentage = (
            (self.eye_contact_frames / self.total_frames) * 100
            if self.total_frames > 0 else 0
        ) 
        eye_contact_time = self.eye_contact_frames / self.fps if self.fps > 0 else 0

        
        total_time = time.time() - self.start_time

        blink_rate = (self.blink_count / total_time) * 60 if total_time > 0 else 0

        avg_closure = np.mean(self.closure_durations) if self.closure_durations else 0
        avg_ibi = np.mean(self.ibi_values) if self.ibi_values else 0
        ibi_std = np.std(self.ibi_values) if self.ibi_values else 0

        longest_no_blink = max(self.ibi_values) if self.ibi_values else 0

        avg_movement = np.mean(self.movement_history) if self.movement_history else 0
        gaze_variance = np.var(self.movement_history) if self.movement_history else 0

        
        if self.fixation_frame_counter > 0:
            fixation_duration = self.fixation_frame_counter / self.fps
            if fixation_duration >= self.FIXATION_MIN_DURATION:
                self.fixation_count += 1
                self.fixation_durations.append(fixation_duration)
        
        avg_fixation = np.mean(self.fixation_durations) if self.fixation_durations else 0
        return {
            "eye_contact_time": eye_contact_time,
            "eye_contact_percentage": eye_contact_percentage,
            "duration": total_time,
            "blink_rate": blink_rate,
            "avg_closure_duration": avg_closure,
            "avg_IBI": avg_ibi,
            "IBI_std": ibi_std,
            "long_closures": self.long_closure_count,
            "total_fixations": self.fixation_count,
            "avg_fixation_duration": avg_fixation,
            "avg_movement": avg_movement,
            "gaze_variance": gaze_variance,
            "longest_no_blink": longest_no_blink
        }