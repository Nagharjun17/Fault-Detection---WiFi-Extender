import os, time, argparse, signal, sys, json, ssl
import numpy as np
import cv2
import onnxruntime as ort
import paho.mqtt.client as mqtt

p = argparse.ArgumentParser(description="LED watcher over RTSP")
p.add_argument("--url", default="rtsp://127.0.0.1:8554/test",
               help="RTSP URL from SmartCam (default: rtsp://127.0.0.1:8554/test)")
p.add_argument("--backend", choices=["ffmpeg", "gst"], default="ffmpeg",
               help="OpenCV backend for RTSP (ffmpeg or gst)")
p.add_argument("--model", default=os.path.expanduser("~/mobilenetv2_led.onnx"),
               help="Path to ONNX model")
p.add_argument("--img-size", type=int, default=224,
               help="Model input size (square). Use what you trained with.")
p.add_argument("--threshold", type=float, default=0.80,
               help="Probability threshold for LED_ON")
p.add_argument("--debounce", type=float, default=0.8,
               help="Seconds the state must persist before reporting a change")
p.add_argument("--print-every", type=float, default=1.0,
               help="Seconds between status prints")
args = p.parse_args()

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

if not os.path.exists(args.model):
    sys.stderr.write(f"ERROR: model not found: {args.model}\n")
    sys.exit(1)

providers = ["CPUExecutionProvider"]
sess = ort.InferenceSession(args.model, providers=providers)
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

CLASS_NAMES = ["LED_OFF", "LED_ON"]

def preprocess_bgr(bgr, size):
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = bgr[y0:y0+side, x0:x0+side]
    img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))[None, ...]
    return chw

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def infer(frame):
    x = preprocess_bgr(frame, args.img_size)
    logits = sess.run([out_name], {inp_name: x})[0]
    probs = softmax(logits[0])
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def open_capture():
    url = args.url
    if args.backend == "ffmpeg":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000|max_delay;5000000"
        )
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    gst = (
        "rtspsrc location=%s protocols=tcp latency=200 "
        "do-rtsp-keep-alive=true tcp-timeout=5000000 retry=3 ! "
        "rtpjitterbuffer drop-on-latency=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=2"
    ) % url
    return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

def reconnecting_capture():
    cap = open_capture()
    backoff = 0.5
    while True:
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(backoff)
        backoff = min(backoff * 1.7, 5.0)
        cap = open_capture()

MQTT_HOST = os.getenv("MQTT_HOST", "")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASS = os.getenv("MQTT_PASS", "")
MQTT_TOPIC_STATE = os.getenv("MQTT_TOPIC_STATE", "kv260/ledwatch/state")
MQTT_TOPIC_PROBS = os.getenv("MQTT_TOPIC_PROBS", "kv260/ledwatch/probs")

def mk_mqtt():
    if not (MQTT_HOST and MQTT_USER and MQTT_PASS):
        print("MQTT disabled (missing env). Set MQTT_HOST, MQTT_USER, MQTT_PASS to enable.")
        return None
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="kv260-ledwatch")
    c.username_pw_set(MQTT_USER, MQTT_PASS)
    c.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    c.tls_insecure_set(False)
    c.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    c.loop_start()
    return c

def mqtt_publish(c, topic, payload, qos=0, retain=False):
    if c is None: 
        return
    try:
        c.publish(topic, json.dumps(payload), qos=qos, retain=retain)
    except Exception as e:
        print(f"MQTT publish error: {e}")

STOP = False
def _sigint(_sig, _frm):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, _sigint)
signal.signal(signal.SIGTERM, _sigint)

def main():
    mqttc = mk_mqtt()
    cap = reconnecting_capture()
    last_state = None
    last_change_at = time.time()
    last_print = 0.0

    while not STOP:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            cap = reconnecting_capture()
            continue

        label, p, probs = infer(frame)

        now = time.time()
        confident_on = (label == "LED_ON" and p >= args.threshold)
        current_state = "LED_ON" if confident_on else "LED_OFF"

        if last_state is None:
            last_state = current_state
            last_change_at = now

        if current_state != last_state:
            if (now - last_change_at) >= args.debounce:
                last_state = current_state
                last_change_at = now
                print(f"[{time.strftime('%H:%M:%S')}] State changed -> {last_state} (p={p:.2f})")
                mqtt_publish(mqttc, MQTT_TOPIC_STATE,
                             {"state": last_state, "p": round(p,3), "ts": int(now)},
                             qos=1, retain=True)
        else:
            last_change_at = now

        if (now - last_print) >= args.print_every:
            print(f"[{time.strftime('%H:%M:%S')}] State: {last_state} (p={p:.2f})")
            mqtt_publish(mqttc, MQTT_TOPIC_PROBS,
                         {"on": round(float(probs[1]),3),
                          "off": round(float(probs[0]),3),
                          "ts": int(now)},
                         qos=0, retain=False)
            last_print = now

    cap.release()

if __name__ == "__main__":
    main()
