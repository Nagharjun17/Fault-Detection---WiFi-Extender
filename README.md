# Fault Detection on WiFi Extender – KV260 SmartCam

Simple end‑to‑end pipeline to detect whether an LED is **ON** or **OFF** using a MobileNetV2 classifier and the Kria KV260 SmartCam RTSP stream, with optional MQTT publishing.

---

## 1. Repository layout

- `train_led.py`       – Train MobileNetV2 LED classifier on your dataset.
- `export_onnx.py`     – Export the best checkpoint to a single‑file ONNX model.
- `ledwatch.py`        – Run ONNX inference on an RTSP stream and publish state via MQTT.
- `make_dataset.sh`    – Helper script to build the `dataset/` folder (optional).
- `dataset/`           – Image dataset (`train` / `val` with `LED_OFF` / `LED_ON` folders).
- `models/`            – Trained checkpoints and exported ONNX (`mobilenetv2_led*.pth/.onnx`).
- `videos/`, `tmp_frames/` – Optional local debugging assets.
- `requirements.txt`   – Python dependencies.

You can run training + export on your laptop, and runtime inference on the Kria board.

---

## 2. Host (laptop) setup – training + ONNX export

### 2.1 Create a Python env and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Prepare the dataset

Organize images as:

```text
dataset/
  train/
    LED_OFF/*.jpg
    LED_ON/*.jpg
  val/
    LED_OFF/*.jpg
    LED_ON/*.jpg
```

You can also use `make_dataset.sh` if you already capture frames from video.

### 2.3 Train the LED classifier

From the project root:

```bash
python train_led.py
```

This trains a MobileNetV2 model and saves the best checkpoint to:

```text
models/mobilenetv2_led_best.pth
```

### 2.4 Export to ONNX

```bash
python export_onnx.py
```

This will:

- Load `models/mobilenetv2_led_best.pth`
- Export a single‑file ONNX model:

```text
models/mobilenetv2_led.onnx
```

---

## 3. Copy the ONNX model to the Kria board

From your laptop (replace `<kria-ip>` with the board’s IP):

```bash
scp models/mobilenetv2_led.onnx ubuntu@<kria-ip>:~/mobilenetv2_led.onnx
```

You can also copy `ledwatch.py` and `requirements.txt` the same way if the repo is not already on the board.

---

## 4. Kria KV260 – SmartCam RTSP setup

All commands below run on the **Kria board** (Linux shell).

### 4.1 Load the SmartCam app

```bash
sudo xmutil desktop_disable
sudo xmutil unloadapp
sudo xmutil loadapp kv260-smartcam
```

### 4.2 Start SmartCam Docker and RTSP stream

```bash
sudo docker run   --env="DISPLAY"   -h "xlnx-docker"   --env="XDG_SESSION_TYPE"   --net=host   --privileged   -v $HOME/.Xauthority:/root/.Xauthority:rw   -v /tmp:/tmp   -v /dev:/dev   -v /sys:/sys   -v /etc/vart.conf:/etc/vart.conf   -v /lib/firmware/xilinx:/lib/firmware/xilinx   -v /run:/run   -it xilinx/smartcam:2022.1 bash

cd /opt/xilinx/kv260-smartcam/bin
./smartcam --mipi -W 1920 -H 1080 --target rtsp -R
```

The SmartCam app will print RTSP URLs such as:

```text
rtsp://172.17.0.1:554/test
rtsp://192.168.1.75:554/test
```

Pick the IP that is reachable from where `ledwatch.py` will run (on the board, `192.168.x.x` is usually what you want).

Leave this Docker container running.

---

## 5. Kria KV260 – LED watcher runtime

These steps are for the same board, but **outside** the SmartCam Docker container.

### 5.1 Create a Python env and install deps

```bash
python3 -m venv ledwatch-venv
source ledwatch-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure `ledwatch.py` and `mobilenetv2_led.onnx` are present in your home directory (or any directory you prefer).

### 5.2 Optional: configure MQTT

Create a file called `.ledwatch.env` in your home directory:

```bash
MQTT_HOST=0af63598be3b461b8f02906b70afe579.s1.eu.hivemq.cloud
MQTT_PORT=8883
MQTT_USER=fpgacv
MQTT_PASS=Fpga1129

MQTT_TOPIC_STATE=kv260/ledwatch/state
MQTT_TOPIC_PROBS=kv260/ledwatch/probs
```

Then load it before running `ledwatch.py`:

```bash
set -a
source .ledwatch.env
set +a
```

If these variables are not set, MQTT is simply disabled and the script will just print to stdout.

### 5.3 Run LED watcher

From the directory containing `ledwatch.py` and `mobilenetv2_led.onnx`:

```bash
source ledwatch-venv/bin/activate

python ledwatch.py   --url rtsp://192.168.1.75:554/test   --backend ffmpeg   --model ./mobilenetv2_led.onnx   --img-size 224   --threshold 0.80   --debounce 0.8   --print-every 1.0
```

You should see lines like:

```text
[00:28:20] State: LED_ON (p=1.00)
[00:28:31] State changed -> LED_OFF (p=0.79)
```

If MQTT is configured, state changes and probabilities are also published to:

- `kv260/ledwatch/state`
- `kv260/ledwatch/probs`

---

## 6. Summary

1. **Train + export on laptop**: `python train_led.py` → `python export_onnx.py`.
2. **Copy ONNX to Kria**: `scp models/mobilenetv2_led.onnx ubuntu@<kria-ip>:~/`.
3. **Start SmartCam RTSP** on Kria via Docker.
4. **Create venv and install deps** on Kria.
5. **Run `ledwatch.py`** pointing to the RTSP URL and ONNX model; optionally enable MQTT.

This is enough to go from raw camera feed on the KV260 to real‑time LED ON/OFF detection and MQTT updates.
