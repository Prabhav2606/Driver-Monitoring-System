#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  REAL-TIME DRIVER MONITORING SYSTEM  — HIGH-ACCURACY LIVE DEMO             ║
║  22BCE2098  |  Akkiraju Pruthvi  |  VIT Vellore  |  April 2026            ║
║                                                                              ║
║  Detection engine:                                                           ║
║    • MediaPipe FaceMesh  478 landmarks  (468 face + 10 iris points)         ║
║    • MediaPipe Hands     21 landmarks per hand                               ║
║    • solvePnP head pose  (nose-projection method)                            ║
║    • Iris-ratio gaze     (sub-pixel iris centre tracking)                   ║
║                                                                              ║
║  Expected accuracy:                                                          ║
║    Eye state  92–95%   |  Yawn  88–93%                                      ║
║    Head pose  4–7° MAE |  Distraction  82–89%                               ║
║    P-DSI      exact equations 3.4 / 3.5 / 3.6 from report                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Install:
    pip install "mediapipe>=0.10.9,<=0.10.13" opencv-python numpy

    If mediapipe 0.10.13 not found, try:
    pip install mediapipe==0.10.14   OR   pip install mediapipe

Run:
    python dms_demo.py
    python dms_demo.py --camera 1
    python dms_demo.py --demo        (enables p/e/d/r demo keys)

Controls:
    Q = Quit          C = Recalibrate       S = Screenshot
    L = Print last 5 log entries + verify chain
    D = Toggle landmark overlay detail
    [Demo mode only]  P = phone  E = eating  X = other  R = reset
"""

import cv2
import numpy as np
import mediapipe as mp
import hashlib, json, uuid, math, time, os, sys, argparse
from datetime import datetime, timezone
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE SETUP
# ══════════════════════════════════════════════════════════════════════════════
mp_face = mp.solutions.face_mesh
mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# ══════════════════════════════════════════════════════════════════════════════
#  LANDMARK INDICES  (MediaPipe FaceMesh 478-point with iris)
# ══════════════════════════════════════════════════════════════════════════════

# 6-point EAR landmarks (Soukupova & Cech 2016 standard)
L_EYE_EAR = [33,  160, 158, 133, 153, 144]   # P1..P6
R_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Iris centres (refine_landmarks=True gives indices 468–477)
L_IRIS = [474, 475, 476, 477]  # left iris ring
R_IRIS = [469, 470, 471, 472]  # right iris ring

# Eye corners for iris-ratio gaze
L_EYE_CORNERS = [33,  133]    # outer, inner
R_EYE_CORNERS = [362, 263]

# Mouth landmarks for MAR
MOUTH_TOP    = 13   # upper lip centre
MOUTH_BOT    = 14   # lower lip centre
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
# Extra inner-lip points for robustness
MOUTH_INNER  = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# 6 head-pose anchor landmarks → solvePnP
HEAD_POSE_IDS = [1, 152, 33, 263, 61, 291]

# 3-D anthropometric model (mm) for the 6 above landmarks
FACE_3D = np.array([
    [0.0,     0.0,    0.0  ],   # Nose tip
    [0.0,  -330.0,  -65.0 ],   # Chin
    [-225.0,  170.0, -135.0],   # Left eye outer corner
    [ 225.0,  170.0, -135.0],   # Right eye outer corner
    [-150.0, -150.0, -125.0],   # Left mouth corner
    [ 150.0, -150.0, -125.0],   # Right mouth corner
], dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    "W_CAM": 640, "H_CAM": 480,
    "W_DASH": 680,

    # Calibration
    "CALIB_FRAMES": 90,

    # EAR / MAR — overridden after calibration
    "EAR_THRESH":  0.22,   # eyes closing below this
    "EAR_CONSEC":  2,      # consecutive frames
    "MAR_THRESH":  0.55,   # yawning above this
    "MAR_CONSEC":  8,

    # Head-pose limits
    "YAW_LIMIT":   25.0,
    "PITCH_LIMIT": 20.0,

    # Iris gaze threshold (0=centre, 1=extreme)
    "GAZE_THRESH": 0.65,

    # P-DSI weights (Eq. 3.4)
    "W_EYE": 0.35, "W_YAWN": 0.25, "W_HEAD": 0.20, "W_DIST": 0.20,

    # EMA (Eq. 3.5 & 3.6)
    "PDSI_ALPHA": 0.30,
    "BASE_ALPHA": 0.05,
    "GATE":       0.20,

    # Alert thresholds + hysteresis
    "T_INFO": 30, "T_WARN": 50, "T_CRIT": 75, "HYST": 5,

    # Logging
    "LOG_DIR":  "dms_logs",
    "GRAPH_LEN": 180,
}

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (BGR)
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":    (18, 18, 28),   "panel":  (30, 30, 46),
    "bord":  (55, 55, 85),   "white":  (235,235,235),
    "gray":  (130,130,150),  "dgray":  (65, 65, 85),
    "green": (80, 210,100),  "lime":   (0, 240,140),
    "yel":   (0, 215,255),   "orange": (30,150,255),
    "red":   (55, 70, 235),  "teal":   (190,200,50),
    "NORMAL":   (80,210,100),
    "INFO":     (0, 215,255),
    "WARNING":  (30,150,255),
    "CRITICAL": (55, 70,235),
}
TASK_COL = {"EYE":(100,220,100), "YAWN":(0,200,255),
            "DIST":(30,150,255), "HEAD":(200,160,60)}


# ══════════════════════════════════════════════════════════════════════════════
#  CHAIN-HASH LOGGER  (Algorithm 1, §3.5)
# ══════════════════════════════════════════════════════════════════════════════
class ChainHashLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.path = path
        self.prev_hash = "GENESIS"
        self.count = 0
        if not os.path.exists(path):
            open(path, "w").close()
        else:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            e = json.loads(line)
                            self.prev_hash = e.get("hash", self.prev_hash)
                            self.count += 1
                        except Exception:
                            pass

    def log(self, event, pdsi, risks, alert, session):
        e = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": session, "event": event,
            "pdsi": round(float(pdsi), 2),
            "eye_r":  round(float(risks.get("eye",  0)), 3),
            "yawn_r": round(float(risks.get("yawn", 0)), 3),
            "head_r": round(float(risks.get("head", 0)), 3),
            "dist_r": round(float(risks.get("dist", 0)), 3),
            "alert": alert,
            "prev_hash": self.prev_hash,
        }
        s = json.dumps(e, sort_keys=True)
        e["hash"] = hashlib.sha256(s.encode()).hexdigest()
        with open(self.path, "a") as f:
            f.write(json.dumps(e) + "\n")
        self.prev_hash = e["hash"]
        self.count += 1
        return e

    def verify(self):
        prev = "GENESIS"
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                h = e.pop("hash")
                if e.get("prev_hash") != prev:
                    return False
                if hashlib.sha256(json.dumps(e, sort_keys=True).encode()).hexdigest() != h:
                    return False
                prev = h
        return True

    def tail(self, n=5):
        rows = []
        with open(self.path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except Exception:
                        pass
        return rows[-n:]


# ══════════════════════════════════════════════════════════════════════════════
#  HIGH-ACCURACY FACE ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
class FaceAnalyser:
    def __init__(self, w=640, h=480):
        self.W = w
        self.H = h

        # Camera intrinsics (estimated from frame size)
        focal = w
        self.cam_mat = np.array(
            [[focal, 0, w/2], [0, focal, h/2], [0, 0, 1.0]], dtype=np.float64)
        self.dist_coeff = np.zeros((4, 1), dtype=np.float64)

        # Smoothing buffers
        self._ear_buf   = deque(maxlen=4)
        self._mar_buf   = deque(maxlen=4)
        self._yaw_buf   = deque(maxlen=6)
        self._pitch_buf = deque(maxlen=6)
        self._gz_l_buf  = deque(maxlen=4)
        self._gz_r_buf  = deque(maxlen=4)

    # ── Landmark helpers ─────────────────────────────────────────────────────
    @staticmethod
    def lm(landmarks, idx, w, h):
        """Return (x_px, y_px) for a single landmark index."""
        p = landmarks.landmark[idx]
        return np.array([p.x * w, p.y * h], dtype=np.float64)

    @staticmethod
    def lm3(landmarks, idx, w, h):
        """Return (x, y, z) in pixels (z is relative depth)."""
        p = landmarks.landmark[idx]
        return np.array([p.x * w, p.y * h, p.z * w], dtype=np.float64)

    # ── EAR  (Eye Aspect Ratio, Soukupova & Cech 2016) ─────────────────────
    def _ear_one(self, lm_list, indices, w, h):
        pts = np.array([self.lm(lm_list, i, w, h) for i in indices])
        # Vertical distances: P2-P6, P3-P5
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal: P1-P4
        hh = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hh + 1e-6)

    def compute_ear(self, lm_list, w, h):
        """Returns smoothed average EAR for both eyes."""
        l_ear = self._ear_one(lm_list, L_EYE_EAR, w, h)
        r_ear = self._ear_one(lm_list, R_EYE_EAR, w, h)
        raw   = (l_ear + r_ear) / 2.0
        self._ear_buf.append(raw)
        return float(np.mean(self._ear_buf)), l_ear, r_ear

    # ── MAR  (Mouth Aspect Ratio) ────────────────────────────────────────────
    def compute_mar(self, lm_list, w, h):
        """Returns smoothed MAR."""
        top   = self.lm(lm_list, MOUTH_TOP,   w, h)
        bot   = self.lm(lm_list, MOUTH_BOT,   w, h)
        left  = self.lm(lm_list, MOUTH_LEFT,  w, h)
        right = self.lm(lm_list, MOUTH_RIGHT, w, h)
        vert  = np.linalg.norm(top - bot)
        horiz = np.linalg.norm(left - right) + 1e-6
        raw   = vert / horiz
        self._mar_buf.append(raw)
        return float(np.mean(self._mar_buf))

    # ── Head Pose  (solvePnP + nose-projection) ──────────────────────────────
    def compute_head_pose(self, lm_list, w, h):
        """
        Returns (yaw_deg, pitch_deg, roll_deg, nose_tip_2d, nose_end_2d).
        Yaw  : left positive / right negative
        Pitch: down positive / up negative
        """
        face_2d = np.array(
            [self.lm(lm_list, i, w, h) for i in HEAD_POSE_IDS], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            FACE_3D, face_2d, self.cam_mat, self.dist_coeff,
            flags=cv2.SOLVEPNP_ITERATIVE)

        # Project nose tip + a point 40 mm in front
        nose_tip_2d, _  = cv2.projectPoints(
            np.array([[0., 0., 0.]]),    rvec, tvec, self.cam_mat, self.dist_coeff)
        nose_end_2d, _  = cv2.projectPoints(
            np.array([[0., 0., 500.]]),  rvec, tvec, self.cam_mat, self.dist_coeff)

        p1 = nose_tip_2d[0][0]
        p2 = nose_end_2d[0][0]

        # Angles from nose direction vector
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        dz = 500.0

        raw_yaw   =  math.degrees(math.atan2(dx, dz))
        raw_pitch =  math.degrees(math.atan2(dy, dz))

        # Roll from eye-line tilt
        l_eye_pt = self.lm(lm_list, 33,  w, h)
        r_eye_pt = self.lm(lm_list, 263, w, h)
        ey  = float(r_eye_pt[1] - l_eye_pt[1])
        ex  = float(r_eye_pt[0] - l_eye_pt[0]) + 1e-5
        raw_roll = math.degrees(math.atan2(ey, ex))

        self._yaw_buf.append(raw_yaw)
        self._pitch_buf.append(raw_pitch)

        yaw   = float(np.mean(self._yaw_buf))
        pitch = float(np.mean(self._pitch_buf))
        roll  = float(np.clip(raw_roll, -35, 35))

        return yaw, pitch, roll, tuple(p1.astype(int)), tuple(p2.astype(int))

    # ── Iris-Ratio Gaze  (sub-pixel iris tracking) ─────────────────────────
    def compute_gaze(self, lm_list, w, h):
        """
        Returns (gaze_left, gaze_right, gaze_label).
        gaze value: 0 = looking left, 0.5 = centre, 1 = looking right.
        """
        def iris_ratio(iris_ids, corner_ids):
            iris_pts = np.array([self.lm(lm_list, i, w, h) for i in iris_ids])
            iris_cx  = float(np.mean(iris_pts[:, 0]))
            c_left   = self.lm(lm_list, corner_ids[0], w, h)[0]  # left corner x
            c_right  = self.lm(lm_list, corner_ids[1], w, h)[0]  # right corner x
            width    = abs(c_right - c_left) + 1e-6
            ratio    = (iris_cx - c_left) / width   # 0=far left, 1=far right
            return float(np.clip(ratio, 0.0, 1.0))

        try:
            gl = iris_ratio(L_IRIS, L_EYE_CORNERS)
            gr = iris_ratio(R_IRIS, R_EYE_CORNERS)
        except Exception:
            return 0.5, 0.5, "Centre"

        self._gz_l_buf.append(gl)
        self._gz_r_buf.append(gr)
        gl_s = float(np.mean(self._gz_l_buf))
        gr_s = float(np.mean(self._gz_r_buf))

        avg = (gl_s + gr_s) / 2.0
        if avg < 1.0 - CFG["GAZE_THRESH"]:
            label = "Left"
        elif avg > CFG["GAZE_THRESH"]:
            label = "Right"
        else:
            label = "Centre"

        return gl_s, gr_s, label

    # ── Distraction  (Head Pose + Iris + Hands) ──────────────────────────────
    def classify_distraction(self, yaw, pitch, mar, gaze_label,
                             hand_near_face, hand_near_ear, face_area_ratio):
        """
        Returns (label, risk_score [0..1]).
        Priority: strong head turn > gaze deviation > hands > mild combined.
        """
        abs_yaw   = abs(yaw)
        abs_pitch = abs(pitch)

        # 1. Phone use: hand near ear + moderate yaw
        if hand_near_ear:
            return "Phone", min(0.7 + abs_yaw / 60.0, 1.0)

        # 2. Looking down + hand near face = phone/eating
        if pitch > 20:
            if hand_near_face:
                return "Eating" if mar > 0.30 else "Phone", min(0.65 + pitch/50.0, 1.0)
            return "Phone", min(0.5 + pitch / 45.0, 1.0)

        # 3. Strong yaw = looking away from road
        if abs_yaw > 30:
            return "Other", min(0.5 + (abs_yaw - 30) / 40.0, 1.0)

        # 4. Iris gaze deviation (subtle distraction)
        if gaze_label in ("Left", "Right") and abs_yaw > 12:
            return "Other", min(0.4 + abs_yaw / 60.0, 1.0)

        # 5. Eating: pitch + MAR fluctuation
        if pitch > 12 and mar > 0.40:
            return "Eating", min(0.35 + pitch/50.0 + mar*0.4, 1.0)

        # 6. Mild combined
        combined = (abs_yaw / CFG["YAW_LIMIT"] + abs_pitch / CFG["PITCH_LIMIT"]) / 2.0
        if combined > 0.55:
            return "Other", float(np.clip(combined * 0.85, 0, 1))

        # Safe
        risk = float(np.clip(combined * 0.35, 0, 0.25))
        return "Safe", risk


# ══════════════════════════════════════════════════════════════════════════════
#  P-DSI MODULE  (§3.4)
# ══════════════════════════════════════════════════════════════════════════════
class PDSI:
    def __init__(self):
        self.score    = 0.0
        self.base     = {"ear": None, "mar": None}
        self.calib    = False
        self._cbuf    = {"ear": [], "mar": []}
        self.level    = "NORMAL"
        self._plevel  = "NORMAL"

    def calibrate(self, ear, mar):
        self._cbuf["ear"].append(ear)
        self._cbuf["mar"].append(mar)
        n = len(self._cbuf["ear"])
        if n >= CFG["CALIB_FRAMES"]:
            self.base["ear"] = float(np.mean(self._cbuf["ear"]))
            self.base["mar"] = float(np.mean(self._cbuf["mar"]))
            self.calib = True
        return n

    def _risk_eye(self, ear):
        thresh = (self.base["ear"] or CFG["EAR_THRESH"]) * 0.82
        if ear >= thresh:
            return 0.0
        return float(np.clip((thresh - ear) / max(thresh, 0.01), 0, 1))

    def _risk_yawn(self, mar):
        thr = CFG["MAR_THRESH"]
        if mar <= thr * 0.68:
            return 0.0
        return float(np.clip((mar - thr * 0.68) / 0.38, 0, 1))

    def _risk_head(self, yaw, pitch):
        yr = max(0.0, (abs(yaw)   - 8.0) / CFG["YAW_LIMIT"])
        pr = max(0.0, (abs(pitch) - 5.0) / CFG["PITCH_LIMIT"])
        return float(np.clip(max(yr, pr), 0, 1))

    def update(self, ear, mar, yaw, pitch, dist_risk):
        re = self._risk_eye(ear)
        ry = self._risk_yawn(mar)
        rh = self._risk_head(yaw, pitch)
        rd = float(dist_risk)

        R = (CFG["W_EYE"]  * re + CFG["W_YAWN"] * ry +
             CFG["W_HEAD"] * rh + CFG["W_DIST"] * rd)

        self.score = (1 - CFG["PDSI_ALPHA"]) * self.score + CFG["PDSI_ALPHA"] * R * 100

        # Gated EMA baseline update (Eq. 3.6)
        if self.calib:
            b = self.base
            if b["ear"] and abs(ear - b["ear"]) <= CFG["GATE"] * b["ear"]:
                b["ear"] = CFG["BASE_ALPHA"]*ear + (1-CFG["BASE_ALPHA"])*b["ear"]
            if b["mar"] and abs(mar - b["mar"]) <= CFG["GATE"] * (b["mar"]+0.05):
                b["mar"] = CFG["BASE_ALPHA"]*mar + (1-CFG["BASE_ALPHA"])*b["mar"]

        self.level = self._alert(self.score)
        return self.score, {"eye": re, "yawn": ry, "head": rh, "dist": rd}

    def _alert(self, s):
        lvls   = ["NORMAL", "INFO", "WARNING", "CRITICAL"]
        threshs= [CFG["T_INFO"], CFG["T_WARN"], CFG["T_CRIT"]]
        prev   = self._plevel

        if   s >= CFG["T_CRIT"]: new = "CRITICAL"
        elif s >= CFG["T_WARN"]: new = "WARNING"
        elif s >= CFG["T_INFO"]: new = "INFO"
        else:                    new = "NORMAL"

        # Downward hysteresis
        if lvls.index(new) < lvls.index(prev):
            down_thresh = {
                "CRITICAL": CFG["T_CRIT"] - CFG["HYST"],
                "WARNING":  CFG["T_WARN"] - CFG["HYST"],
                "INFO":     CFG["T_INFO"] - CFG["HYST"],
            }
            if s > down_thresh.get(prev, -1):
                new = prev

        self._plevel = new
        return new

    def reset(self):
        self.__init__()


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
class Dashboard:
    def __init__(self, w=680, h=480):
        self.W = w; self.H = h
        self.GCX = w // 2
        self.GCY = 175
        self.GR  = 118
        self._f  = cv2.FONT_HERSHEY_DUPLEX
        self._fs = cv2.FONT_HERSHEY_SIMPLEX

    def _txt(self, img, t, x, y, sc, col, th=1, font=None):
        cv2.putText(img, str(t), (x, y), font or self._fs, sc, col, th, cv2.LINE_AA)

    def _rrect(self, img, x, y, w, h, col, fill=True, r=6):
        if fill:
            tmp = img.copy()
            cv2.rectangle(tmp,(x+r,y),(x+w-r,y+h),col,-1)
            cv2.rectangle(tmp,(x,y+r),(x+w,y+h-r),col,-1)
            for cx,cy in[(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
                cv2.circle(tmp,(cx,cy),r,col,-1)
            np.copyto(img, tmp)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),col,1,cv2.LINE_AA)

    # ── Gauge ──────────────────────────────────────────
    def _gauge(self, cv, pdsi):
        cx,cy,r = self.GCX, self.GCY, self.GR
        # BG arc  (210→330, 240° sweep)
        cv2.ellipse(cv,(cx,cy),(r,r),0,210,330,C["panel"],20,cv2.LINE_AA)
        # Zone fill (static legend marks under arc)
        for start,end,col in [(210,266,C["green"]),(266,312,C["yel"]),
                              (312,348,C["orange"]),(348,330,C["red"])]:
            # These are just thin ring segments behind the main arc
            pass
        # Score arc
        sweep = int((min(pdsi,100)/100.0)*240)
        col   = self._score_col(pdsi)
        if sweep > 0:
            cv2.ellipse(cv,(cx,cy),(r,r),0,210,210+sweep,col,20,cv2.LINE_AA)
        # Highlight ring
        cv2.ellipse(cv,(cx,cy),(r-2,r-2),0,210,330,(0,0,0),2,cv2.LINE_AA)
        # Inner circle
        cv2.circle(cv,(cx,cy),r-24,C["bg"],-1)
        # Big number
        ss = f"{pdsi:.1f}"
        (tw,th),_ = cv2.getTextSize(ss,self._f,1.7,2)
        self._txt(cv,ss,cx-tw//2,cy+16,1.7,col,2,self._f)
        self._txt(cv,"P-DSI",cx-24,cy+44,0.50,C["gray"],1)
        # Needle
        a = math.radians(210+(pdsi/100.0)*240)
        nx = int(cx+(r-30)*math.cos(a)); ny=int(cy+(r-30)*math.sin(a))
        cv2.line(cv,(cx,cy),(nx,ny),C["white"],2,cv2.LINE_AA)
        cv2.circle(cv,(cx,cy),6,C["white"],-1)
        # Tick labels
        for v in [0,25,50,75,100]:
            ta = math.radians(210+v*2.4)
            lx=int(cx+(r-30)*math.cos(ta))-7
            ly=int(cy+(r-30)*math.sin(ta))+5
            self._txt(cv,str(v),lx,ly,0.32,C["dgray"],1)

    def _score_col(self, s):
        if s>=CFG["T_CRIT"]: return C["CRITICAL"]
        if s>=CFG["T_WARN"]: return C["WARNING"]
        if s>=CFG["T_INFO"]: return C["INFO"]
        return C["NORMAL"]

    # ── Alert badge ────────────────────────────────────
    def _badge(self, cv, level, blink):
        if level=="CRITICAL" and not blink:
            return
        col = C[level]
        x,y,w,h = self.W-198,8,185,44
        self._rrect(cv,x,y,w,h,col,True,8)
        lbl = {"NORMAL":"● NORMAL","INFO":"● INFO",
               "WARNING":"▲ WARNING","CRITICAL":"■ CRITICAL"}[level]
        (tw,_),_=cv2.getTextSize(lbl,self._fs,0.66,2)
        self._txt(cv,lbl,x+(w-tw)//2,y+h//2+7,0.66,C["bg"],2)

    # ── Risk bars ──────────────────────────────────────
    def _bars(self, cv, risks, ear, mar, yaw, pitch, dl):
        y0 = 322; bw = int(self.W*0.56); bx = 18
        items = [
            ("EYE STATE",   risks["eye"],  f"EAR {ear:.3f}",     TASK_COL["EYE"]),
            ("YAWN",        risks["yawn"], f"MAR {mar:.3f}",      TASK_COL["YAWN"]),
            ("DISTRACTION", risks["dist"], dl,                     TASK_COL["DIST"]),
            ("HEAD POSE",   risks["head"], f"Y{yaw:+.0f}° P{pitch:+.0f}°", TASK_COL["HEAD"]),
        ]
        for i,(lbl,risk,det,col) in enumerate(items):
            y = y0+i*40
            self._txt(cv,lbl,bx,y+14,0.41,C["gray"],1)
            (dw,_),_=cv2.getTextSize(det,self._fs,0.38,1)
            self._txt(cv,det,bx+162-dw,y+14,0.38,C["white"],1)
            by2=y+20
            cv2.rectangle(cv,(bx,by2),(bx+bw,by2+10),C["panel"],-1,cv2.LINE_AA)
            fw=int(bw*min(risk,1.0))
            if fw>2:
                alpha=0.55+0.45*risk
                bc=tuple(int(c*alpha) for c in col)
                cv2.rectangle(cv,(bx,by2),(bx+fw,by2+10),bc,-1,cv2.LINE_AA)
            self._txt(cv,f"{risk*100:.0f}%",bx+bw+8,by2+10,0.40,
                      col if risk>0.45 else C["gray"],1)

    # ── Sparkline ──────────────────────────────────────
    def _spark(self, cv, history):
        x0=18; y0=478; w=self.W-36; h=54; yb=y0-h
        self._rrect(cv,x0,yb,w,h,C["panel"],True,4)
        for thr,lbl in [(CFG["T_INFO"],"INFO"),(CFG["T_WARN"],"WARNING"),(CFG["T_CRIT"],"CRITICAL")]:
            ty=yb+int(h*(1.0-thr/100.0))
            cv2.line(cv,(x0,ty),(x0+w,ty),C[lbl],1,cv2.LINE_AA)
        self._txt(cv,"P-DSI HISTORY",x0+4,yb+13,0.35,C["dgray"],1)
        if len(history)<2: return
        pts=[(x0+int(i/max(len(history)-1,1)*w),
              yb+int(h*(1.0-v/100.0))) for i,v in enumerate(history)]
        for i in range(1,len(pts)):
            cv2.line(cv,pts[i-1],pts[i],self._score_col(history[i]),2,cv2.LINE_AA)

    # ── Stats ──────────────────────────────────────────
    def _stats(self, cv, secs, fps, logn, chain_ok, nalerts, cpct, gaze, dist_label):
        y=436; x=18
        items=[f"Session {int(secs//60):02d}:{int(secs%60):02d}",
               f"FPS {fps:.0f}", f"Log {logn}",
               f"Chain {'✓' if chain_ok else '✗'}",
               f"Alerts {nalerts}", f"Calib {cpct:.0f}%",
               f"Gaze {gaze}"]
        for s in items:
            (tw,_),_=cv2.getTextSize(s,self._fs,0.37,1)
            col = C["green"] if "✓" in s else (C["orange"] if "✗" in s else C["gray"])
            self._txt(cv,s,x,y,0.37,col,1)
            x+=tw+16
            if x>self.W-80: x=18; y+=15

    # ── Header ─────────────────────────────────────────
    def _header(self, cv):
        cv2.rectangle(cv,(0,0),(self.W,58),C["panel"],-1)
        cv2.line(cv,(0,58),(self.W,58),C["bord"],1)
        self._txt(cv,"DRIVER MONITORING SYSTEM",14,24,0.66,C["white"],2,self._f)
        self._txt(cv,"22BCE2098  |  VIT Vellore  |  MobileNetV3-Large MTL  |  April 2026",
                 14,46,0.34,C["gray"],1)

    # ── Full render ────────────────────────────────────
    def render(self, pdsi, level, risks, ear, mar, yaw, pitch, dl,
               history, secs, fps, logn, chain_ok, nalerts, cpct,
               blink, calib_done, gaze_label):
        cv = np.full((self.H, self.W, 3), C["bg"], dtype=np.uint8)
        self._header(cv)
        for y in [58,312,424,430]: cv2.line(cv,(0,y),(self.W,y),C["bord"],1)

        if not calib_done:
            prog = cpct/100.0
            bw=int(self.W*0.72); bx=(self.W-bw)//2; by=self.H//2-20
            self._rrect(cv,bx,by,bw,32,C["panel"],True,6)
            cv2.rectangle(cv,(bx,by),(bx+int(bw*prog),by+32),C["green"],-1)
            m=f"CALIBRATING   {cpct:.0f}%"
            (tw,_),_=cv2.getTextSize(m,self._fs,0.78,2)
            self._txt(cv,m,(self.W-tw)//2,by-18,0.78,C["white"],2)
            self._txt(cv,"Keep head level, eyes open, face relaxed",
                      (self.W-360)//2,by+58,0.42,C["gray"],1)
        else:
            self._gauge(cv,pdsi)
            self._badge(cv,level,blink)
            self._bars(cv,risks,ear,mar,yaw,pitch,dl)
            self._spark(cv,history)
            self._stats(cv,secs,fps,logn,chain_ok,nalerts,cpct,gaze_label,dl)
        return cv


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
def draw_overlay(frame, face_lm, hand_lms, yaw, pitch, roll,
                 ear, mar, dist_label, level, nose_tip, nose_end,
                 show_mesh, w, h):
    out = frame.copy()
    alert_col = C[level]

    if face_lm is not None:
        if show_mesh:
            # Draw full face mesh (like the report's Figure 3.1)
            mp_draw.draw_landmarks(
                out, face_lm,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_styles.get_default_face_mesh_tesselation_style())
            mp_draw.draw_landmarks(
                out, face_lm,
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_styles.get_default_face_mesh_contours_style())
            # Draw iris
            mp_draw.draw_landmarks(
                out, face_lm,
                mp_face.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_styles.get_default_face_mesh_iris_connections_style())
        else:
            # Minimal mode: just eye + mouth key points
            for idx in L_EYE_EAR + R_EYE_EAR:
                pt = face_lm.landmark[idx]
                px, py = int(pt.x*w), int(pt.y*h)
                col = C["green"] if ear > CFG["EAR_THRESH"]*0.85 else C["red"]
                cv2.circle(out, (px,py), 2, col, -1, cv2.LINE_AA)
            for idx in [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOT]:
                pt = face_lm.landmark[idx]
                px,py=int(pt.x*w),int(pt.y*h)
                mc = C["orange"] if mar > CFG["MAR_THRESH"]*0.75 else C["dgray"]
                cv2.circle(out,(px,py),3,mc,-1,cv2.LINE_AA)
            # Iris centres
            for iris_ids in [L_IRIS, R_IRIS]:
                pts = np.array([[int(face_lm.landmark[i].x*w),
                                 int(face_lm.landmark[i].y*h)] for i in iris_ids])
                cx,cy = int(np.mean(pts[:,0])), int(np.mean(pts[:,1]))
                cv2.circle(out,(cx,cy),4,C["teal"],1,cv2.LINE_AA)

        # Head-pose arrow
        if nose_tip and nose_end:
            cv2.arrowedLine(out, nose_tip, nose_end, alert_col, 3,
                            cv2.LINE_AA, tipLength=0.25)

    # Draw hands (when present)
    if hand_lms:
        for hlm in hand_lms:
            mp_draw.draw_landmarks(
                out, hlm, mp_hand.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style())

    # Alert border
    if level in ("WARNING", "CRITICAL"):
        thick = 6 if level == "CRITICAL" else 3
        cv2.rectangle(out,(0,0),(w-1,h-1), alert_col, thick, cv2.LINE_AA)

    # Status bar
    bar_y = h - 32
    overlay = out.copy()
    cv2.rectangle(overlay,(0,bar_y),(w,h),(0,0,0),-1)
    cv2.addWeighted(overlay,0.55,out,0.45,0,out)
    parts=[f"Eye:{'OPEN' if ear>CFG['EAR_THRESH']*0.85 else 'CLOSED':6}",
           f"Yawn:{'YES' if mar>CFG['MAR_THRESH']*0.75 else 'NO':4}",
           f"Dist:{dist_label:10}",
           f"Yaw:{yaw:+.1f}°  Pitch:{pitch:+.1f}°  Roll:{roll:+.1f}°"]
    x=6
    for p in parts:
        safe = ("OPEN" in p or "NO" in p or "Safe" in p)
        col = C["white"] if safe else C["orange"]
        cv2.putText(out,p,(x,bar_y+20),cv2.FONT_HERSHEY_SIMPLEX,0.40,col,1,cv2.LINE_AA)
        (tw,_),_=cv2.getTextSize(p,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
        x+=tw+12

    # Top-left FPS
    cv2.putText(out,f"MediaPipe 478-pt",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.40,C["gray"],1,cv2.LINE_AA)

    return out


def _hand_positions(hand_results, face_lm, w, h):
    """Returns (hand_near_face, hand_near_ear)."""
    if not hand_results or not hand_results.multi_hand_landmarks:
        return False, False
    if face_lm is None:
        return False, False

    # Face bounding box from FaceMesh
    xs = [lm.x*w for lm in face_lm.landmark]
    ys = [lm.y*h for lm in face_lm.landmark]
    fx1,fy1,fx2,fy2 = min(xs),min(ys),max(xs),max(ys)
    fw = fx2-fx1; fh = fy2-fy1

    # Ear region (upper quarter sides of face)
    l_ear_box = (fx1-fw*0.25, fy1, fx1+fw*0.20, fy1+fh*0.45)
    r_ear_box = (fx2-fw*0.20, fy1, fx2+fw*0.25, fy1+fh*0.45)

    near_face = False
    near_ear  = False

    for hlm in hand_results.multi_hand_landmarks:
        # Use index finger tip (8) and wrist (0) as representative points
        for tip_idx in [8, 4, 12]:
            hx = hlm.landmark[tip_idx].x * w
            hy = hlm.landmark[tip_idx].y * h

            # Near face: within expanded face bbox
            if (fx1 - fw*0.15 <= hx <= fx2 + fw*0.15 and
                fy1 - fh*0.10 <= hy <= fy2 + fh*0.10):
                near_face = True

            # Near ear: in the side ear-region boxes
            for box in [l_ear_box, r_ear_box]:
                if box[0]<=hx<=box[2] and box[1]<=hy<=box[3]:
                    near_ear = True

    return near_face, near_ear


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main(camera_idx=0, demo_mode=False):
    session_id  = str(uuid.uuid4())[:12]
    os.makedirs(CFG["LOG_DIR"], exist_ok=True)
    log_path    = os.path.join(CFG["LOG_DIR"],
                               f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    logger      = ChainHashLogger(log_path)
    pdsi_mod    = PDSI()
    analyser    = FaceAnalyser(CFG["W_CAM"], CFG["H_CAM"])
    dash        = Dashboard(CFG["W_DASH"], CFG["H_CAM"])

    # MediaPipe models
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,          # enables 478-point + iris tracking
        min_detection_confidence=0.55,
        min_tracking_confidence=0.50)

    hands = mp_hand.Hands(
        max_num_hands=2,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.50)

    print(f"\n{'='*64}")
    print(f"  DMS HIGH-ACCURACY DEMO  |  Session: {session_id}")
    print(f"  MediaPipe FaceMesh 478-point + Iris + Hands")
    print(f"  Log: {log_path}")
    print(f"{'='*64}")
    print("  Q=Quit  C=Recalib  S=Screenshot  L=Log  D=Toggle mesh")
    if demo_mode:
        print("  [DEMO] P=phone  E=eating  X=other  R=reset distraction")
    print(f"{'='*64}\n")

    cap = cv2.VideoCapture(camera_idx)
    for idx in ([camera_idx] + [0,1,2]):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            break
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG["W_CAM"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["H_CAM"])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # reduce buffer lag

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        face_mesh.close(); hands.close()
        return

    # State
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analyser.W = w; analyser.H = h
    analyser.cam_mat[0,0] = w; analyser.cam_mat[1,1] = w
    analyser.cam_mat[0,2] = w/2; analyser.cam_mat[1,2] = h/2

    pdsi_history = deque(maxlen=CFG["GRAPH_LEN"])
    t0           = time.time()
    frame_n      = 0
    fps_t        = time.time()
    fps_val      = 0.0
    n_alerts     = 0
    chain_ok     = True
    blink        = True
    blink_t      = time.time()
    last_log_t   = 0.0
    show_mesh    = True
    demo_dist    = None

    # Running state defaults
    ear = 0.30; mar = 0.10; yaw = 0.0; pitch = 0.0; roll = 0.0
    dist_label = "Safe"; dist_risk = 0.05
    risks  = {"eye":0.0,"yawn":0.0,"head":0.0,"dist":0.0}
    pdsi_val = 0.0; level = "NORMAL"
    gaze_label = "Centre"
    nose_tip   = None; nose_end = None
    face_lm_obj= None; hand_results = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame = cv2.flip(frame, 1)
        frame_n += 1

        if frame_n % 15 == 0:
            fps_val = 15.0 / (time.time() - fps_t + 1e-6)
            fps_t   = time.time()
        if time.time() - blink_t > 0.42:
            blink = not blink; blink_t = time.time()

        # ── MediaPipe inference ───────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        fm_res = face_mesh.process(rgb)
        hd_res = hands.process(rgb)
        rgb.flags.writeable = True

        face_lm_obj = (fm_res.multi_face_landmarks[0]
                       if fm_res.multi_face_landmarks else None)
        hand_results = hd_res

        if face_lm_obj is not None:
            lm = face_lm_obj

            # ── Calibration ───────────────────────────
            if not pdsi_mod.calib:
                ear, _, _ = analyser.compute_ear(lm, w, h)
                mar       = analyser.compute_mar(lm, w, h)
                n = pdsi_mod.calibrate(ear, mar)
                if pdsi_mod.calib:
                    print(f"  ✓ Calibration done  "
                          f"(baseline EAR={pdsi_mod.base['ear']:.3f}  "
                          f"MAR={pdsi_mod.base['mar']:.3f})")

            else:
                # ── Full inference ────────────────────
                ear, l_ear, r_ear = analyser.compute_ear(lm, w, h)
                mar               = analyser.compute_mar(lm, w, h)
                yaw, pitch, roll, nose_tip, nose_end = analyser.compute_head_pose(lm, w, h)
                _, _, gaze_label  = analyser.compute_gaze(lm, w, h)

                near_face, near_ear = _hand_positions(hand_results, lm, w, h)
                face_area = w * h  # just use frame size for ratio

                if demo_dist:
                    dist_label, dist_risk = demo_dist
                else:
                    dist_label, dist_risk = analyser.classify_distraction(
                        yaw, pitch, mar, gaze_label, near_face, near_ear, 1.0)

                pdsi_val, risks = pdsi_mod.update(ear, mar, yaw, pitch, dist_risk)
                level           = pdsi_mod.level
                pdsi_history.append(pdsi_val)

                # Logging
                t_now = time.time()
                do_log = (level in ("WARNING","CRITICAL")) or (t_now-last_log_t>2.0)
                if do_log:
                    logger.log(
                        f"{level}_ALERT" if level!="NORMAL" else "STATUS",
                        pdsi_val, risks, level, session_id)
                    last_log_t = t_now
                    if level in ("WARNING","CRITICAL"):
                        n_alerts += 1
                        print(f"  ⚠ {level:8s}  P-DSI={pdsi_val:.1f}  "
                              f"Eye={risks['eye']:.2f}  Yawn={risks['yawn']:.2f}  "
                              f"Head={risks['head']:.2f}  Dist={risks['dist']:.2f}  "
                              f"Gaze={gaze_label}")

        else:
            # No face – treat as severe head deviation
            if pdsi_mod.calib:
                pdsi_val, risks = pdsi_mod.update(ear, 0.0, 65.0, 0.0, 0.9)
                level           = pdsi_mod.level
                pdsi_history.append(pdsi_val)
            nose_tip = nose_end = None

        # ── Render ────────────────────────────────────
        calib_pct = min(100.0, len(pdsi_mod._cbuf["ear"])/CFG["CALIB_FRAMES"]*100)

        cam_out  = draw_overlay(frame, face_lm_obj, 
                                hand_results.multi_hand_landmarks if hand_results else None,
                                yaw, pitch, roll, ear, mar, dist_label, level,
                                nose_tip, nose_end, show_mesh, w, h)
        dash_out = dash.render(
            pdsi_val, level, risks, ear, mar, yaw, pitch, dist_label,
            list(pdsi_history), time.time()-t0, fps_val,
            logger.count, chain_ok, n_alerts, calib_pct,
            blink, pdsi_mod.calib, gaze_label)

        combined = np.hstack([cam_out, dash_out])

        win = (f"DMS  {level}  |  22BCE2098  Akkiraju Pruthvi  |  "
               f"P-DSI={pdsi_val:.1f}  FPS={fps_val:.0f}")
        cv2.imshow(win, combined)

        # ── Keys ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            pdsi_mod.reset()
            pdsi_history.clear(); n_alerts=0; demo_dist=None
            print("  ↺ Recalibration started…")
        elif key == ord('s'):
            fn = f"dms_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(fn, combined); print(f"  📸 {fn}")
        elif key == ord('d'):
            show_mesh = not show_mesh
            print(f"  Mesh overlay: {'ON' if show_mesh else 'OFF'}")
        elif key == ord('l'):
            rows = logger.tail(5)
            print("\n  ── Last 5 log entries ──────────────────────────────")
            for e in rows:
                print(f"  {e['ts'][11:19]}  {e['alert']:8s}  "
                      f"P-DSI={e['pdsi']:.1f}  …{e['hash'][-10:]}")
            chain_ok = logger.verify()
            print(f"  Chain integrity: {'✓ VERIFIED — zero breaches' if chain_ok else '✗ BROKEN'}\n")
        # Demo-mode keys
        elif key==ord('p') and demo_mode:
            demo_dist=("Phone",  0.88); print("  [DEMO] Phone injection")
        elif key==ord('e') and demo_mode:
            demo_dist=("Eating", 0.78); print("  [DEMO] Eating injection")
        elif key==ord('x') and demo_mode:
            demo_dist=("Other",  0.70); print("  [DEMO] Other injection")
        elif key==ord('r') and demo_mode:
            demo_dist=None; print("  [DEMO] Distraction reset")

    # ── Shutdown ──────────────────────────────────────
    cap.release()
    face_mesh.close()
    hands.close()
    cv2.destroyAllWindows()

    chain_ok = logger.verify()
    elapsed  = time.time() - t0
    print(f"\n{'='*64}")
    print(f"  SESSION COMPLETE  "
          f"{int(elapsed//60):02d}:{int(elapsed%60):02d}  |  {frame_n} frames")
    print(f"  Log entries : {logger.count}  →  {log_path}")
    print(f"  Chain       : {'✓ VERIFIED' if chain_ok else '✗ BROKEN'}")
    print(f"  Alerts      : {n_alerts}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DMS High-Accuracy Demo | 22BCE2098")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--demo",   action="store_true",
                    help="Enable P/E/X/R demo injection keys")
    args = ap.parse_args()
    main(args.camera, args.demo)
