"""
full_pipeline_v1.py  —  COMPLETE MUTED VIDEO → AI SOUND

THREE LAYERS IN ONE PIPELINE:
  Layer 1 — EMOTION MUSIC     (face expression → happy/sad/angry/fear music)
  Layer 2 — ENVIRONMENT BG    (BLIP location → beach/traffic/kitchen ambient)
  Layer 3 — OBJECT SOUNDS     (BLIP objects + motion → dog bark/door knock/footstep)

VOLUMES:
  Emotion music  : 30% (full music) or 15% (ambient texture)
  Environment BG :  8% (always subtle)
  Object sounds  : 75% (placed at exact onset frames)

RUN:
  & "project_env/Scripts/python.exe" full_pipeline_v1.py --video input1.mp4 --steps 150
  & "project_env/Scripts/python.exe" full_pipeline_v1.py --video input1.mp4 --desc "dog barking" --steps 150
  & "project_env/Scripts/python.exe" full_pipeline_v1.py --video input1.mp4 --scene dog --steps 150
"""

import os, gc, sys, subprocess, argparse
import cv2, torch, numpy as np, soundfile as sf
from PIL import Image
from tqdm import tqdm
from scipy import signal as sg
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import warnings; warnings.filterwarnings("ignore")

try:
    import trampoline
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "trampoline"])
    os.execv(sys.executable, [sys.executable] + sys.argv)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def gpu_info():
    if torch.cuda.is_available():
        n = torch.cuda.get_device_name(0)
        t = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {n}  {t:.1f}GB")


# ═══════════════════════════════════════════════════════════════════
#  SCENE KEYWORDS  (object detection from BLIP captions)
# ═══════════════════════════════════════════════════════════════════
SCENE_KEYWORDS = [
    ("dog",        ["dog","puppy","canine","hound","labrador","retriever","poodle",
                    "bulldog","shepherd","beagle","husky","terrier","corgi"]),
    ("cat",        ["cat","kitten","feline","kitty","tabby","persian","siamese"]),
    ("bird",       ["bird","sparrow","robin","pigeon","parrot","crow","eagle",
                    "hawk","owl","dove","duck","goose","chirping","avian"]),
    ("horse",      ["horse","pony","stallion","mare","foal","equine","mustang"]),
    ("cow",        ["cow","bull","cattle","calf","buffalo","bison"]),
    ("lion",       ["lion","tiger","leopard","cheetah","panther","jaguar"]),
    ("wolf",       ["wolf","fox","coyote","howling"]),
    ("elephant",   ["elephant","mammoth","trumpeting"]),
    ("monkey",     ["monkey","ape","gorilla","chimpanzee","baboon"]),
    ("bear",       ["bear","grizzly","panda"]),
    ("thunder",    ["thunder","lightning","thunderstorm","storm cloud"]),
    ("fire",       ["fire","flame","burning","blaze","campfire","fireplace"]),
    ("rain",       ["rain","rainy","drizzle","downpour","raining"]),
    ("ocean",      ["ocean","sea","waves","wave","surf","beach","shore"]),
    ("running",    ["running","jogging","sprinting","runner"]),
    ("walking",    ["walking","walk","strolling","stepping","footstep"]),
    ("clapping",   ["clap","clapping","applause","cheering","applauding"]),
    ("laughing",   ["laugh","laughing","giggling","laughter"]),
    ("baby",       ["baby","infant","newborn","toddler","child crying"]),
    ("cooking",    ["cooking","frying","sizzle","stove","pan","grill",
                    "bowl of food","plate of food","eating","meal","kitchen cooking"]),
    ("chopping",   ["chop","chopping","knife","cutting board","slicing"]),
    ("coffee",     ["coffee","espresso","latte","mug","cup of coffee"]),
    ("kettle",     ["kettle","teapot","whistle","boiling","tea"]),
    ("train",      ["train","railway","locomotive","metro","subway","platform"]),
    ("car",        ["car","vehicle","automobile","driving","traffic"]),
    ("guitar",     ["guitar","strumming","acoustic guitar","playing guitar"]),
    ("piano",      ["piano","grand piano","pianist","piano keys"]),
    ("drums",      ["drum","drumming","percussion","snare","drum kit"]),
    ("basketball", ["basketball","hoop","dribble","court","bouncing a ball"]),
    ("swimming",   ["swimming","pool","diving","splash","swimmer"]),
    ("gym",        ["gym","exercise","workout","weights","lifting"]),
    ("phone",      ["rotary","dial","telephone","handset","landline"]),
    ("keyboard",   ["keyboard","typing","computer","laptop","typewriter"]),
    ("gun",        ["gunshot","gun","pistol","rifle","shooting"]),
    ("crowd",      ["crowd","audience","spectators","stadium","concert","many people"]),
    ("door",       ["knock","knocking","door knock","opening a door",
                    "open a door","a door","the door","front door"]),
]

def get_scene(text):
    cl = text.lower()
    for scene, kws in SCENE_KEYWORDS:
        for kw in kws:
            if kw in cl:
                return scene
    return "default"


# ═══════════════════════════════════════════════════════════════════
#  OBJECT SOUND DEFINITIONS
#  (bg_env_hint, bg_label, event_prompt, event_label, event_dur_s)
# ═══════════════════════════════════════════════════════════════════
SCENE_SOUNDS = {
    "dog":        ("outdoor park, gentle breeze", "Park",
                   "single sharp dog bark, one bark, dry, close microphone, no reverb",
                   "Dog Bark", 0.8),
    "cat":        ("quiet cozy indoor", "Home",
                   "single cat meow, sharp, dry, close microphone", "Cat Meow", 0.8),
    "bird":       ("outdoor garden, morning", "Garden",
                   "single bird chirp, sharp tweet, dry, close microphone", "Bird Chirp", 0.5),
    "horse":      ("open field, light wind", "Field",
                   "single horse hoof strike, hard ground, close microphone", "Hoofbeat", 0.3),
    "cow":        ("quiet farm", "Farm",
                   "single cow moo, clear, close microphone", "Cow Moo", 1.0),
    "lion":       ("open savanna", "Savanna",
                   "single lion roar, deep powerful, dry, close microphone", "Lion Roar", 1.5),
    "wolf":       ("quiet night forest", "Night",
                   "wolf howl, single long howl, close microphone", "Wolf Howl", 1.5),
    "elephant":   ("open bush", "Bush",
                   "elephant trumpet, single call, close microphone", "Elephant Call", 1.2),
    "monkey":     ("tropical jungle", "Jungle",
                   "monkey screech, single call, close microphone", "Monkey Call", 0.8),
    "bear":       ("pine forest", "Forest",
                   "bear growl, single deep growl, dry, close microphone", "Bear Growl", 1.0),
    "thunder":    ("overcast sky, pre-storm", "Storm",
                   "thunder crack, single powerful boom, close microphone", "Thunder", 1.5),
    "fire":       ("warm indoor", "Indoor",
                   "fire crackle, wood pop, sharp dry, close microphone", "Fire Crackle", 0.6),
    "rain":       ("rainy outdoor", "Rain",
                   "rain burst, heavy downpour on surface, close microphone", "Rain Burst", 1.0),
    "ocean":      ("coastal", "Coast",
                   "wave crash, large wave on shore, close microphone", "Wave Crash", 1.5),
    "running":    ("indoor corridor", "Corridor",
                   "single running footstep, hard floor, dry, close microphone", "Running Step", 0.25),
    "walking":    ("indoor room", "Room",
                   "single footstep, heel on hard floor, dry, no reverb, close microphone",
                   "Footstep", 0.3),
    "clapping":   ("auditorium", "Auditorium",
                   "single hand clap, crisp, dry, close microphone", "Clap", 0.25),
    "laughing":   ("living room", "Living Room",
                   "laugh burst, genuine laughter, close microphone", "Laughter", 0.8),
    "baby":       ("nursery", "Nursery",
                   "baby cry, infant wail, sharp cry, close microphone", "Baby Cry", 1.2),
    "cooking":    ("kitchen, ventilation fan", "Kitchen",
                   "sizzle burst, oil in hot pan, close microphone", "Sizzle", 0.8),
    "chopping":   ("kitchen", "Kitchen",
                   "knife chop, single on cutting board, dry, close microphone", "Chop", 0.3),
    "coffee":     ("morning kitchen", "Kitchen",
                   "coffee pour, liquid into ceramic cup, close microphone", "Coffee Pour", 1.2),
    "kettle":     ("kitchen", "Kitchen",
                   "kettle whistle burst, steam, close microphone", "Kettle Whistle", 0.8),
    "train":      ("train station platform", "Station",
                   "train horn, single loud blast, close microphone", "Train Horn", 1.2),
    "car":        ("city street", "Street",
                   "engine rev, single acceleration, close microphone", "Engine Rev", 1.2),
    "guitar":     ("rehearsal room", "Studio",
                   "guitar strum, single chord, acoustic, dry, close microphone", "Guitar Strum", 1.2),
    "piano":      ("concert hall", "Hall",
                   "piano note, single key, acoustic, close microphone", "Piano Note", 1.2),
    "drums":      ("rehearsal space", "Studio",
                   "drum hit, single snare, sharp, dry, close microphone", "Drum Hit", 0.25),
    "basketball": ("indoor gymnasium", "Gym",
                   "ball bounce, single dribble on hard court, close microphone", "Ball Bounce", 0.3),
    "swimming":   ("indoor pool", "Pool",
                   "water splash, body entering water, close microphone", "Splash", 0.6),
    "gym":        ("gym", "Gym",
                   "weight drop, heavy impact on floor, close microphone", "Weight Drop", 0.4),
    "phone":      ("quiet indoor", "Indoor",
                   "rotary dial click, mechanical ratchet, dry, close microphone", "Dial Click", 0.4),
    "keyboard":   ("quiet office", "Office",
                   "key click, single keypress, mechanical, close microphone", "Key Click", 0.2),
    "gun":        ("outdoor range", "Outdoor",
                   "gunshot, single sharp shot, loud crack, dry, close microphone", "Gunshot", 0.4),
    "crowd":      ("large venue", "Venue",
                   "crowd cheer burst, sudden roar, close microphone", "Crowd Cheer", 1.0),
    "door":       ("indoor hallway", "Hallway",
                   "door knock, knuckle on solid wooden door, hard hollow thud, "
                   "dry, no reverb, close microphone", "Door Knock", 0.4),
    "default":    ("quiet indoor", "Room",
                   "sharp impact sound, burst, dry, close microphone", "Impact", 0.5),
}


# ═══════════════════════════════════════════════════════════════════
#  ENVIRONMENT KEYWORDS + SOUNDS  (location → background ambient)
# ═══════════════════════════════════════════════════════════════════
ENV_KEYWORDS = [
    ("beach",       ["beach","ocean","sea","shore","sand","coastal","surf","seaside"]),
    ("forest",      ["forest","woods","jungle","trees","woodland","nature trail"]),
    ("park",        ["park","garden","outdoor","grass","lawn","field","meadow"]),
    ("traffic",     ["street","road","traffic","highway","city street","urban","downtown"]),
    ("train",       ["train station","railway station","platform","subway station","railroad"]),
    ("airport",     ["airport","terminal","runway","departure","gate"]),
    ("market",      ["market","bazaar","mall","shopping","busy street"]),
    ("construction",["construction","scaffold","crane","building site"]),
    ("office",      ["office","desk","workplace","cubicle","meeting room"]),
    ("kitchen",     ["kitchen","stove","oven","refrigerator","counter"]),
    ("restaurant",  ["restaurant","cafe","diner","dining room","waiter"]),
    ("gym",         ["gym","fitness center","workout room","weight room","treadmill"]),
    ("stadium",     ["stadium","arena","sports field","bleachers","grandstand"]),
    ("rain",        ["rain","rainy","wet","puddle","drizzle","umbrella","raining"]),
    ("mountain",    ["mountain","hill","cliff","valley","peak","rocky"]),
    ("river",       ["river","stream","creek","waterfall","lake","pond"]),
    ("home",        ["living room","bedroom","home","house","apartment",
                     "hallway","couch","sofa","sitting room"]),
]

ENV_SOUNDS = {
    "beach":        ("ocean waves crashing on beach, seagulls, sea breeze, "
                     "coastal atmosphere, distant surf", "Beach Ambient"),
    "forest":       ("deep forest, birds chirping, wind through leaves, "
                     "insects, peaceful woodland", "Forest Ambient"),
    "park":         ("outdoor park, birds singing, light breeze, "
                     "distant children, peaceful garden", "Park Ambient"),
    "traffic":      ("busy city street, cars passing, traffic hum, "
                     "distant horns, urban outdoor", "Traffic Ambient"),
    "train":        ("train station, trains arriving, PA announcements, "
                     "crowd footsteps, railway atmosphere", "Station Ambient"),
    "airport":      ("busy airport terminal, announcements, crowd chatter, "
                     "distant jet engines", "Airport Ambient"),
    "market":       ("busy outdoor market, vendors, crowd chatter, "
                     "lively marketplace", "Market Ambient"),
    "construction": ("construction site, machinery, drilling, hammering, "
                     "workers, building activity", "Construction Ambient"),
    "office":       ("quiet office, computer fans, air conditioning, "
                     "distant keyboard, professional indoor", "Office Ambient"),
    "kitchen":      ("kitchen, ventilation fan, light appliance hum, "
                     "warm cooking atmosphere", "Kitchen Ambient"),
    "restaurant":   ("busy restaurant, cutlery clinking, conversation murmur, "
                     "dining atmosphere", "Restaurant Ambient"),
    "gym":          ("fitness gym, equipment clanking, ventilation, "
                     "distant music, workout atmosphere", "Gym Ambient"),
    "stadium":      ("large stadium, crowd murmur, distant announcer, "
                     "echoing venue", "Stadium Ambient"),
    "rain":         ("heavy rain falling, raindrops on surfaces, "
                     "thunder in distance, stormy weather", "Rain Ambient"),
    "mountain":     ("mountain wind, distant birds, thin air silence, "
                     "high altitude nature", "Mountain Ambient"),
    "river":        ("flowing river over rocks, gentle current, "
                     "birds near water, peaceful stream", "River Ambient"),
    "home":         ("quiet home interior, soft room tone, "
                     "distant outdoor sounds, domestic atmosphere", "Home Ambient"),
    "default":      ("quiet indoor ambient, soft room tone, still air", "Room Tone"),
}

def get_environment(text):
    cl = text.lower()
    for env, kws in ENV_KEYWORDS:
        for kw in kws:
            if kw in cl:
                return env
    return "default"


# ═══════════════════════════════════════════════════════════════════
#  EMOTION CONFIG  (face expression → music)
# ═══════════════════════════════════════════════════════════════════
EMOTION_CONFIG = {
    "happy": {
        "intensity": "high",
        "music_prompt": (
            "upbeat joyful background music, cheerful piano melody, "
            "acoustic guitar, light percussion, major key, 120bpm, "
            "warm bright positive energy, feel-good cinematic soundtrack"),
        "ambient_prompt": None,
        "label": "Happy 😊", "color_bgr": (80, 220, 80),
        "vol_full": 0.30, "vol_ambient": 0.30,
    },
    "sad": {
        "intensity": "low",
        "music_prompt": (
            "slow melancholic piano music, soft cello strings, "
            "minor key, 60bpm, sorrowful and reflective, heartfelt cinematic score"),
        "ambient_prompt": (
            "sad melancholic ambient texture, soft minor key pads, "
            "distant lonely piano, slow atmospheric drone"),
        "label": "Sad 😢", "color_bgr": (200, 120, 80),
        "vol_full": 0.25, "vol_ambient": 0.12,
    },
    "angry": {
        "intensity": "high",
        "music_prompt": (
            "intense aggressive orchestral music, powerful drums, "
            "heavy brass, minor key, 145bpm, dark forceful energy, "
            "action thriller cinematic soundtrack"),
        "ambient_prompt": None,
        "label": "Angry 😠", "color_bgr": (60, 60, 220),
        "vol_full": 0.32, "vol_ambient": 0.32,
    },
    "fear": {
        "intensity": "high",
        "music_prompt": (
            "suspenseful horror music, tense tremolo strings, "
            "dark synth pads, minor key, 65bpm, "
            "psychological thriller atmosphere, tension building"),
        "ambient_prompt": None,
        "label": "Fear 😨", "color_bgr": (60, 200, 200),
        "vol_full": 0.30, "vol_ambient": 0.30,
    },
    "surprise": {
        "intensity": "high",
        "music_prompt": (
            "dramatic cinematic music, sudden orchestral hit, "
            "swelling strings, brass fanfare, major key, "
            "dynamic unexpected momentum, epic reveal"),
        "ambient_prompt": None,
        "label": "Surprise 😲", "color_bgr": (200, 180, 60),
        "vol_full": 0.32, "vol_ambient": 0.32,
    },
    "disgust": {
        "intensity": "low",
        "music_prompt": (
            "unsettling dissonant music, atonal strings, "
            "dark ambient, slow disturbing 55bpm, uneasy tense"),
        "ambient_prompt": (
            "dark unsettling ambient drone, dissonant low frequencies, "
            "uncomfortable pads, eerie quiet atmosphere"),
        "label": "Disgust 🤢", "color_bgr": (80, 140, 180),
        "vol_full": 0.18, "vol_ambient": 0.10,
    },
    "neutral": {
        "intensity": "low",
        "music_prompt": (
            "calm neutral background music, soft ambient piano, "
            "gentle pads, major key, 80bpm, peaceful unobtrusive"),
        "ambient_prompt": (
            "calm neutral ambient texture, soft pad drone, "
            "peaceful gentle atmosphere, barely noticeable"),
        "label": "Neutral 😐", "color_bgr": (160, 160, 160),
        "vol_full": 0.18, "vol_ambient": 0.08,
    },
}
FULL_MUSIC_MIN_S = 3.0


# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — MOTION ANALYSIS (5 zones)
# ═══════════════════════════════════════════════════════════════════
def analyze_motion(video_path):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = total / fps

    print(f"\n[VIDEO]  {os.path.basename(video_path)}")
    print(f"         {total} frames | {fps:.1f}fps | {dur:.2f}s")
    print("[STEP 1] Motion analysis ...")

    ZONES = [
        (0.0, 0.5, 0.0, 0.5), (0.0, 0.5, 0.5, 1.0),
        (0.25, 0.75, 0.25, 0.75),
        (0.5, 1.0, 0.0, 0.5), (0.5, 1.0, 0.5, 1.0),
    ]
    N = len(ZONES)
    zone_raw   = np.zeros((N, total), np.float32)
    prev_zones = [None] * N

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(total):
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        for z, (ys,ye,xs,xe) in enumerate(ZONES):
            crop = frame[int(h*ys):int(h*ye), int(w*xs):int(w*xe)]
            gray = cv2.cvtColor(cv2.resize(crop,(80,60)),
                                cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_zones[z] is not None:
                zone_raw[z,i] = np.mean(np.abs(gray - prev_zones[z]))
            prev_zones[z] = gray
    cap.release()

    for z in range(N):
        mx = zone_raw[z].max()
        if mx > 0: zone_raw[z] /= mx

    raw_n  = zone_raw.max(axis=0)
    smooth = uniform_filter1d(raw_n, size=3)

    peaks, _ = find_peaks(smooth, height=0.28,
                          distance=max(3, int(fps*0.22)), prominence=0.10)
    peaks = [p for p in peaks if p >= int(fps*0.3)]

    baseline = uniform_filter1d(raw_n, size=int(fps*1.0))
    onsets   = []
    for pk in peaks:
        onset = pk
        for j in range(pk-1, max(0, pk-int(fps*0.8)), -1):
            if raw_n[j] <= baseline[j] + 0.04:
                onset = j + 1; break
        onsets.append(onset)

    if len(onsets) >= 2:
        gaps_ms    = [round((onsets[i+1]-onsets[i])/fps*1000)
                      for i in range(len(onsets)-1)]
        min_gap_ms = max(100, min(gaps_ms))
    else:
        min_gap_ms = 800

    calm_frame = int(np.argmin(smooth[:max(1, total//2)]))

    print(f"  Found {len(peaks)} motion events")
    print(f"  min_gap={min_gap_ms}ms  calm_frame=f{calm_frame}")
    return raw_n, smooth, onsets, peaks, min_gap_ms, calm_frame, fps, dur, total


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — SCENE + ENVIRONMENT DETECTION (BLIP)
# ═══════════════════════════════════════════════════════════════════
def detect_scene_and_env(video_path, total, fps, calm_frame, peaks,
                          scene_arg=None, desc_arg=None, n_segments=3):
    print(f"\n[STEP 2] Scene + environment detection ...")

    # Override
    if desc_arg or scene_arg:
        s = _resolve(scene_arg, desc_arg)
        entry = SCENE_SOUNDS.get(s, SCENE_SOUNDS["default"])
        seg_size = total // n_segments
        segments = [(s, entry, "default", i*seg_size,
                     (i+1)*seg_size if i<n_segments-1 else total)
                    for i in range(n_segments)]
        dom_env = "default"
        print(f"  Override → scene=[{s}] env=[default]")
        return segments, s, dom_env

    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from collections import Counter

        proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            use_safetensors=True).to("cuda")
        model.eval()

        vqa_qs = [
            None,
            "Question: What is the person doing? Answer:",
            "Question: What object is being used? Answer:",
            "Question: Where is this taking place? Answer:",
        ]

        seg_size = total // n_segments
        segments = []
        all_envs = []
        cap = cv2.VideoCapture(video_path)

        for i in range(n_segments):
            sf_ = i * seg_size
            ef_ = (i+1)*seg_size if i < n_segments-1 else total
            frames_s = list(dict.fromkeys([
                sf_ + (ef_-sf_)//4, (sf_+ef_)//2,
                sf_ + 3*(ef_-sf_)//4,
            ] + [p for p in peaks if sf_<=p<ef_][:2]))[:4]

            all_caps = []
            for fidx in frames_s:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(fidx, total-1))
                ret, frame = cap.read()
                if not ret: continue
                img = Image.fromarray(
                    cv2.cvtColor(cv2.resize(frame,(384,384)),
                                 cv2.COLOR_BGR2RGB))
                for q in vqa_qs:
                    inp = proc(img, return_tensors="pt") if q is None \
                          else proc(img, text=q, return_tensors="pt")
                    inp = {k: (v.to("cuda", torch.float16)
                               if v.dtype==torch.float32 else v.to("cuda"))
                           for k,v in inp.items()}
                    with torch.no_grad():
                        ids = model.generate(**inp, max_new_tokens=30)
                    ans = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
                    if q: ans = ans.replace(q,"").strip()
                    if ans: all_caps.append(ans)

            sv = [get_scene(c) for c in all_caps]
            ev = [get_environment(c) for c in all_caps]
            sr = [v for v in sv if v!="default"]
            er = [v for v in ev if v!="default"]
            best_s = Counter(sr).most_common(1)[0][0] if sr else "default"
            best_e = Counter(er).most_common(1)[0][0] if er else "default"
            entry  = SCENE_SOUNDS.get(best_s, SCENE_SOUNDS["default"])
            all_envs.append(best_e)
            segments.append((best_s, entry, best_e, sf_, ef_))
            print(f"  Seg {i+1}: scene=[{best_s}] env=[{best_e}] → {entry[3]}")

        cap.release()
        del model, proc; free_gpu()

        sc_counts  = Counter(s[0] for s in segments)
        env_counts = Counter(e for e in all_envs if e!="default")
        dom_scene  = sc_counts.most_common(1)[0][0]
        dom_env    = env_counts.most_common(1)[0][0] if env_counts else "default"
        return segments, dom_scene, dom_env

    except Exception as e:
        print(f"  BLIP failed: {e} → use --desc")
        seg_size = total // n_segments
        segs = [("default", SCENE_SOUNDS["default"], "default",
                 i*seg_size, (i+1)*seg_size if i<n_segments-1 else total)
                for i in range(n_segments)]
        return segs, "default", "default"

def _resolve(scene_arg, desc_arg):
    if desc_arg:
        s = get_scene(desc_arg)
        if s != "default": return s
    if scene_arg:
        s = scene_arg.lower().strip()
        if s in SCENE_SOUNDS: return s
        sc = get_scene(s)
        if sc != "default": return sc
    return "default"


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — EMOTION DETECTION (DeepFace)
# ═══════════════════════════════════════════════════════════════════
def detect_emotions(video_path, total, fps, sample_every=15):
    print(f"\n[STEP 3] Emotion detection ...")
    try:
        from deepface import DeepFace
    except Exception as e:
        print(f"  DeepFace unavailable ({e}) → using neutral for all")
        dur = total / fps
        return [("neutral", 0, total, 0, round(total/fps*1000))]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap     = cv2.VideoCapture(video_path)
    results = []

    for fidx in tqdm(range(0, total, sample_every), desc="  Emotions"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret: continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) == 0:
            results.append((fidx, round(fidx/fps*1000), "neutral")); continue
        try:
            a = DeepFace.analyze(frame, actions=["emotion"],
                                 enforce_detection=False, silent=True)
            if isinstance(a, list): a = a[0]
            raw   = a.get("emotion", {})
            tot_  = sum(raw.values()) or 1.0
            scores = {k.lower(): v/tot_ for k,v in raw.items()}
            dom   = max(scores, key=scores.get)
            results.append((fidx, round(fidx/fps*1000), dom))
        except Exception:
            results.append((fidx, round(fidx/fps*1000), "neutral"))
    cap.release()

    if not results:
        return [("neutral", 0, total, 0, round(total/fps*1000))]

    # Smooth + segment
    from collections import Counter
    seq      = [r[2] for r in results]
    smoothed = []
    W = 5
    for i in range(len(seq)):
        w = seq[max(0,i-W//2):i+W//2+1]
        smoothed.append(Counter(w).most_common(1)[0][0])

    raw_segs = []
    cur=smoothed[0]; cs=0
    for i in range(1, len(smoothed)):
        if smoothed[i] != cur:
            raw_segs.append((cur,cs,i)); cur=smoothed[i]; cs=i
    raw_segs.append((cur,cs,len(smoothed)))

    min_f = max(1, int(2.0*fps/sample_every))
    merged = []
    for emo,s,e in raw_segs:
        if merged and (e-s)<min_f:
            pe,ps,_ = merged[-1]; merged[-1]=(pe,ps,e)
        else:
            merged.append((emo,s,e))

    segments = []
    for emo,si,ei in merged:
        sf_ = results[si][0]
        ef_ = min(results[min(ei,len(results)-1)][0]+sample_every, total)
        sm  = results[si][1]
        em  = round(ef_/fps*1000)
        segments.append((emo,sf_,ef_,sm,em))
    if segments:
        e=segments[-1]
        segments[-1]=(e[0],e[1],total,e[3],round(total/fps*1000))

    from collections import Counter as C2
    counts = C2(r[2] for r in results)
    print(f"  Emotions: {dict(counts.most_common())}")
    print(f"  Segments: {len(segments)}")
    for i,(emo,sf_,ef_,sm,em) in enumerate(segments):
        print(f"    [{i+1}] {emo:10s} {sm}ms–{em}ms")
    return segments


# ═══════════════════════════════════════════════════════════════════
#  AUDIO HELPERS
# ═══════════════════════════════════════════════════════════════════
def trim_silence(audio, sr):
    if len(audio)==0: return audio, 0.0
    mx = np.abs(audio).max()
    if mx<1e-6: return audio, 0.0
    hits = np.where(np.abs(audio)>mx*0.01)[0]
    if len(hits)==0: return audio, 0.0
    onset = max(0, int(hits[0])-int(sr*0.002))
    return audio[onset:], onset/sr*1000

def denoise(audio, sr):
    audio = audio.astype(np.float64)
    audio = sg.sosfilt(sg.butter(4,40,'high',fs=sr,output='sos'),audio)
    audio = sg.sosfilt(sg.butter(4,18000,'low',fs=sr,output='sos'),audio)
    return audio.astype(np.float32)

def make_clip(audio, sr, dur, target=0.88, fade_s=0.008):
    audio = denoise(audio, sr)
    f = int(sr*fade_s)
    if len(audio)>f*2:
        audio[:f]*=np.linspace(0,1,f); audio[-f:]*=np.linspace(1,0,f)
    mx=np.abs(audio).max()
    if mx>0: audio=audio/mx*target
    exact=int(dur*sr)
    return (audio[:exact] if len(audio)>exact
            else np.pad(audio,(0,exact-len(audio)))).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — GENERATE ALL 3 LAYERS
# ═══════════════════════════════════════════════════════════════════
def generate_all_layers(scene_segments, emotion_segments, dom_env,
                         onsets, peaks, raw_n, fps, dur,
                         min_gap_ms, output_path, steps):

    import importlib.util as _ilu
    if _ilu.find_spec("stable_audio_tools") is None:
        for _p in [
            r"C:\Users\admin\muted_video_audio_project\project_env\Lib\site-packages",
            r"C:\Users\admin\Sounds for videos\project_env\Lib\site-packages",
        ]:
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0,_p); break

    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    print(f"\n[STEP 4] Generating 3 audio layers ...")
    model, cfg = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to("cuda").eval()
    sr, ss = cfg["sample_rate"], cfg["sample_size"]
    spf    = sr / fps
    n_samp = int(dur*sr)+sr
    print(f"  sr={sr}Hz  fps={fps:.1f}")

    layer1 = np.zeros(n_samp, np.float32)  # emotion music
    layer2 = np.zeros(n_samp, np.float32)  # environment BG
    layer3 = np.zeros(n_samp, np.float32)  # object sounds

    def gen(prompt, dur_s, n_cand=3, cfg_scale=7.0):
        dur_s = max(min(dur_s, 47.0), 0.5)
        tgt   = int(ss * dur_s / 47.0)
        cond  = [{"prompt": prompt, "seconds_start": 0, "seconds_total": dur_s}]
        best, bsc = None, -9e9
        for _ in range(n_cand):
            with torch.no_grad():
                out = generate_diffusion_cond(
                    model, steps=steps, cfg_scale=cfg_scale,
                    conditioning=cond, sample_size=tgt,
                    sigma_min=0.3, sigma_max=500,
                    sampler_type="dpmpp-3m-sde", device="cuda",
                    seed=int(np.random.randint(0,2**31-1)))
            a  = out[0].cpu().numpy().mean(axis=0)
            fft= np.abs(np.fft.rfft(a[:min(len(a),sr*3)]))
            sc = float(np.sum(fft[80:12000]))
            if sc>bsc: bsc,best=sc,a.copy()
        return best

    # ── LAYER 1: EMOTION MUSIC ───────────────────────────────────
    print(f"\n  [Layer 1] Emotion music ({len(emotion_segments)} segments)")
    emo_cache = {}
    XFADE = 0.5
    for i, (emo, sf_, ef_, sm, em) in enumerate(emotion_segments):
        seg_dur = max((ef_-sf_)/fps, 1.5)
        cfg_e   = EMOTION_CONFIG.get(emo, EMOTION_CONFIG["neutral"])
        inten   = cfg_e["intensity"]
        use_full= (inten=="high" or seg_dur>=FULL_MUSIC_MIN_S
                   or cfg_e["ambient_prompt"] is None)
        prompt  = cfg_e["music_prompt"] if use_full \
                  else (cfg_e["ambient_prompt"] or cfg_e["music_prompt"])
        vol     = cfg_e["vol_full"] if use_full else cfg_e["vol_ambient"]
        key     = f"{emo}_{'full' if use_full else 'amb'}"
        mtype   = "Full" if use_full else "Ambient"
        print(f"    [{i+1}] {emo:10s} {mtype:7s} vol={vol:.2f} "
              f"{sm}ms–{em}ms ({seg_dur:.1f}s)")
        if key in emo_cache:
            raw = emo_cache[key]
        else:
            raw = gen(prompt, min(seg_dur+XFADE*2,47.0),
                      n_cand=3, cfg_scale=7.0)
            emo_cache[key] = raw
            free_gpu()
        clip = make_clip(raw, sr, seg_dur+XFADE*2, fade_s=XFADE)
        start= int(sm/1000*sr); end=min(start+len(clip),n_samp)
        slen = end-start
        if slen<=0: continue
        xf   = int(XFADE*sr)
        env  = np.ones(slen,np.float32)
        if slen>xf*2:
            env[:xf]=np.linspace(0,1,xf); env[-xf:]=np.linspace(1,0,xf)
        layer1[start:end] += clip[:slen]*env*vol

    # ── LAYER 2: ENVIRONMENT BACKGROUND ─────────────────────────
    print(f"\n  [Layer 2] Environment background [{dom_env}]")
    env_entry= ENV_SOUNDS.get(dom_env, ENV_SOUNDS["default"])
    env_p    = env_entry[0]
    print(f"    Prompt: {env_p[:60]}")
    bg_raw   = gen(env_p, min(dur,10.0), n_cand=2, cfg_scale=6.0)
    free_gpu()
    bg_clip  = make_clip(bg_raw, sr, min(dur,10.0))
    reps     = int(np.ceil(n_samp/len(bg_clip)))
    tile     = np.tile(bg_clip, reps)[:n_samp]
    seam     = int(0.08*sr)
    for r in range(1, reps):
        s=r*len(bg_clip)
        if 0<s<n_samp:
            a_=max(0,s-seam); b_=min(n_samp,s+seam)
            tile[a_:s]*=np.linspace(1,0,s-a_)
            tile[s:b_]*=np.linspace(0,1,b_-s)
    layer2 = tile * 0.08   # 8% — very subtle under music

    # ── LAYER 3: OBJECT SOUNDS ───────────────────────────────────
    print(f"\n  [Layer 3] Object sounds ({len(onsets)} events)")
    if onsets:
        obj_cache = {}
        max_clip_s = max(int(min_gap_ms*0.80/1000*sr), int(0.05*sr))

        for i, (onset, peak) in enumerate(zip(onsets, peaks)):
            # Find scene for this onset
            seg_scene = "default"
            seg_entry = SCENE_SOUNDS["default"]
            for sc, ent, env, sf_, ef_ in scene_segments:
                if sf_ <= onset < ef_:
                    seg_scene = sc; seg_entry = ent; break

            ev_p   = seg_entry[2]
            ev_dur = seg_entry[4]
            ev_lab = seg_entry[3]

            if seg_scene not in obj_cache:
                print(f"    Generating [{seg_scene}]: {ev_p[:50]}")
                ev_raw = gen(ev_p, ev_dur, n_cand=5, cfg_scale=9.0)
                free_gpu()
                ev_tr, rm = trim_silence(ev_raw, sr)
                if len(ev_tr) > max_clip_s:
                    ev_cl = ev_tr[:max_clip_s].copy()
                    fade_ = min(int(0.020*sr), max_clip_s//4)
                    ev_cl[-fade_:] *= np.linspace(1,0,fade_)
                else:
                    ev_cl = ev_tr
                obj_cache[seg_scene] = ev_cl

            ev_clip = obj_cache[seg_scene]
            place   = int(onset*spf)
            end     = min(place+len(ev_clip), n_samp)
            slen    = end-place
            if slen<=0: continue
            vol     = 0.50 + float(raw_n[min(peak,len(raw_n)-1)])*0.45
            xf      = min(int(0.006*sr), slen//4)
            env_    = np.ones(slen, np.float32)
            if xf>0:
                env_[:xf]=np.linspace(0,1,xf)
                env_[-xf:]=np.linspace(1,0,xf)
            layer3[place:end] += ev_clip[:slen]*env_*vol
            print(f"    [{i+1:2d}] {seg_scene:12s} f={onset:5d} "
                  f"t={round(onset/fps*1000):6d}ms vol={vol:.2f}")
    else:
        print("    No motion events detected")

    # ── MIX ALL LAYERS ───────────────────────────────────────────
    print(f"\n  Mixing 3 layers ...")
    final = layer1 + layer2 + layer3
    mx    = np.abs(final).max()
    if mx>0: final = final/mx*0.90

    out = final[:int(dur*sr)]
    del model; free_gpu()
    sf.write(output_path, out, sr)
    print(f"  Saved: {output_path}")
    return sr, scene_segments, emotion_segments


# ═══════════════════════════════════════════════════════════════════
#  STEP 5 — BURN CAPTIONS
# ═══════════════════════════════════════════════════════════════════
def burn_captions(video_path, scene_segments, emotion_segments,
                  smooth, fps, total, output_path):
    cap  = cv2.VideoCapture(video_path)
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(output_path,
                           cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))
    font = cv2.FONT_HERSHEY_DUPLEX
    fsc  = max(0.65, W/1440); thk=2
    mean_v = float(np.mean(smooth))

    def get_scene_label(f):
        for sc,ent,env,sf_,ef_ in scene_segments:
            if sf_<=f<ef_: return ent[3] if smooth[min(f,len(smooth)-1)]>mean_v else ent[1]
        return ""

    def get_emo(f):
        for emo,sf_,ef_,sm,em in emotion_segments:
            if sf_<=f<ef_: return emo
        return "neutral"

    print("\n[STEP 5] Burning captions ...")
    fidx=0
    with tqdm(total=total, desc="  frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            emo   = get_emo(fidx)
            cfg_e = EMOTION_CONFIG.get(emo, EMOTION_CONFIG["neutral"])
            color = cfg_e["color_bgr"]
            elabel= cfg_e["label"]
            slabel= get_scene_label(fidx)

            # Top-right: emotion badge
            tw = cv2.getTextSize(elabel,font,fsc*0.75,thk)[0][0]
            bx = W-tw-24; by=44
            ov=frame.copy()
            cv2.rectangle(ov,(bx-10,by-28),(bx+tw+10,by+8),(0,0,0),-1)
            cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
            cv2.putText(frame,elabel,(bx,by),font,fsc*0.75,color,thk,cv2.LINE_AA)

            # Bottom center: object sound label
            if slabel:
                sw=cv2.getTextSize(slabel,font,fsc*0.7,thk)[0][0]
                sx=(W-sw)//2; sy=H-22
                ov2=frame.copy()
                cv2.rectangle(ov2,(sx-10,sy-26),(sx+sw+10,sy+6),(0,0,0),-1)
                cv2.addWeighted(ov2,0.55,frame,0.45,0,frame)
                cv2.putText(frame,slabel,(sx,sy),font,fsc*0.7,
                            (255,255,255),thk,cv2.LINE_AA)

            out.write(frame); fidx+=1; pbar.update(1)
    cap.release(); out.release(); print("  Done.")


# ═══════════════════════════════════════════════════════════════════
#  STEP 6 — MERGE
# ═══════════════════════════════════════════════════════════════════
def merge(video_path, audio_path, out_path):
    import shutil
    if not shutil.which("ffmpeg"): print("[WARN] ffmpeg not found"); return
    print(f"\n[STEP 6] Merging → {out_path}")
    r = subprocess.run([
        "ffmpeg","-y","-i",video_path,"-i",audio_path,
        "-c:v","libx264","-preset","fast","-crf","23",
        "-c:a","aac","-b:a","192k",
        "-map","0:v:0","-map","1:a:0","-shortest",out_path
    ], capture_output=True, text=True)
    ok = r.returncode==0
    print(f"  [{'OK' if ok else 'ERROR'}] {out_path}")
    if not ok: print(r.stderr[-400:])


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",        required=True)
    ap.add_argument("--desc",         default=None,
                    help='Describe objects: "dog barking" / "person knocking on door"')
    ap.add_argument("--scene",        default=None,
                    help="Scene: dog cat door phone walking cooking rain drums etc.")
    ap.add_argument("--segments",     type=int, default=3)
    ap.add_argument("--sample_every", type=int, default=15,
                    help="Emotion sample rate in frames (default 15)")
    ap.add_argument("--output_audio", default="generated_audio.wav")
    ap.add_argument("--output_video", default="output_full.mp4")
    ap.add_argument("--steps",        type=int, default=150)
    args = ap.parse_args()

    print("\n" + "═"*62)
    print("  full_pipeline_v1.py  —  COMPLETE VIDEO → AI SOUND")
    print("  Object sounds + Environment BG + Emotion music")
    print("═"*62)
    gpu_info()

    if not os.path.exists(args.video):
        print(f"[ERROR] Not found: {args.video}"); return

    tmp = "_full_tmp.mp4"

    # Steps 1-3: analysis
    raw_n, smooth, onsets, peaks, min_gap_ms, calm_frame, fps, dur, total = \
        analyze_motion(args.video)

    scene_segments, dom_scene, dom_env = detect_scene_and_env(
        args.video, total, fps, calm_frame, peaks,
        scene_arg=args.scene, desc_arg=args.desc,
        n_segments=args.segments)

    emotion_segments = detect_emotions(
        args.video, total, fps, sample_every=args.sample_every)

    # Print full plan
    print(f"\n[PLAN]")
    print(f"  Video      : {os.path.basename(args.video)} ({dur:.1f}s {fps:.0f}fps)")
    print(f"  Layer 1    : Emotion music ({len(emotion_segments)} segments)")
    for i,(emo,sf_,ef_,sm,em) in enumerate(emotion_segments):
        cfg_e = EMOTION_CONFIG.get(emo, EMOTION_CONFIG["neutral"])
        print(f"    [{i+1}] {emo:10s} {sm}ms–{em}ms → {cfg_e['label']}")
    print(f"  Layer 2    : Environment [{dom_env}] "
          f"— {ENV_SOUNDS.get(dom_env,ENV_SOUNDS['default'])[1]}")
    print(f"  Layer 3    : Object sounds ({len(onsets)} events)")
    for i,(sc,ent,env,sf_,ef_) in enumerate(scene_segments):
        print(f"    Seg {i+1}: [{sc}] → {ent[3]}")

    # Step 4: generate
    generate_all_layers(
        scene_segments, emotion_segments, dom_env,
        onsets, peaks, raw_n, fps, dur,
        min_gap_ms, args.output_audio, args.steps)

    # Step 5: captions
    burn_captions(args.video, scene_segments, emotion_segments,
                  smooth, fps, total, tmp)

    # Step 6: merge
    merge(tmp, args.output_audio, args.output_video)
    if os.path.exists(tmp): os.remove(tmp)

    print("\n" + "═"*62)
    print("  DONE!")
    print(f"  Audio : {args.output_audio}")
    print(f"  Video : {args.output_video}")
    print("═"*62 + "\n")


if __name__ == "__main__":
    main()