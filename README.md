# X-Guard: oneM2M foreign-object inspection for a conveyor

X-Guard checks products on a conveyor belt for foreign objects and pushes bad ones off the line.
An Arduino with two IR sensors and an MG996R servo watches the belt, a gateway connects it to an
ACME MN-CSE, a camera AE uploads a photo of each product, a YOLOv8n model (meant to run on MEC)
judges it, and the rejector and a Streamlit dashboard act on the result.

None of the programs call each other directly. Each one writes to the MN-CSE and gets what it
needs through subscription notifications. The data formats follow our "X-Guard final system spec".

```
arduino/                  hardware team's Arduino code (2026-11-15): S1/S2 E18-D80NK, queue, MG996R rejector
  src/main.cpp            PlatformIO build (same content as xguard_2sensor.ino)
  xguard_2sensor/         Arduino IDE sketch
cse/acme.ini              ACME MN-CSE config (MQTT on)
cse/init/xguard.fcp       flexContainer specialization xg:inRet
setup/setup_resources.py  creates the AEs, containers, flexContainer, group, ACPs and subscriptions
common/m2m.py             shared oneM2M HTTP helpers, reads .env
gateway/gateway.py        serial <-> MQTT (LineSensor-01, Rejector-01). SERIAL_PORT=SIM runs a fake Arduino
gateway/image_source.py   on each arrival, uploads a camera frame (OV9281, 850 nm) or a dataset image (ImageSource-01, :9200)
ai/app.py                 image notification -> models/detect.pt -> verdict (Inspection, :9000)
ai/Dockerfile             build from the repo root
train/                    YOLOv8n training (data in train/dataset -> models/detect.pt)
train/colab_train.ipynb   the training notebook we actually used (Colab GPU, data on Drive)
models/                   trained model, classes.txt, model_info.json
mec/                      k8s Deployment and NodePort 30900
dashboard/app.py          Streamlit UI (:8501) plus a notification receiver (:9100)
.streamlit/config.toml    dashboard theme
tests/requests.http       hand-written requests for the VS Code REST Client
dataset/                  demo images for IMAGE_SOURCE=dataset (sorted by name = seq order)
.env                      addresses, ports and model settings in one place
```

Each product goes through these steps, and the logs number them the same way:

1. The Arduino sees the product at S1 and sends `DET,seq,0`.
2. The gateway writes it to `LineSensor-01/arrivals`.
3. The image AE is notified of the arrival.
4. It uploads the photo to `ImageSource-01/xrayImages`.
5. The AI is notified of the image.
6. It writes the verdict to `Inspection/inspectionResult` (and to `defects` on FAIL).
7. The gateway (as Rejector-01) is notified of the verdict.
8. It sends `OK,seq` or `NG,seq` to the Arduino.
9. When the product reaches S2 the Arduino passes or rejects it, and the gateway logs that to `rejectLog`.
10. The dashboard gets the defect notification and reads everything else from the CSE.

## Setup (Windows, VS Code terminal)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # only if the AI runs on this PC
```

Install Mosquitto and leave it running as a Windows service on the default port 1883. The
recommended VS Code extensions are listed in `vscode-settings/extensions.json`.

## Running everything

Double-click `run_all.bat`, or run `python run_all.py` (add `--reset` to start with an empty CSE database).
It starts things in order: the MQTT broker (amqtt if Mosquitto isn't running), ACME, the base
resources, the AI, image AE and dashboard, the subscriptions, and finally the gateway. Then it
opens the dashboard in your browser. Logs go to `logs/`. Ctrl+C stops everything.

If `models/detect.pt` is missing, the AI starts in dummy mode and judges by file name.

## Training the model

We trained on a Colab GPU with `train/colab_train.ipynb`.

1. Put the data zips in a Google Drive folder, e.g. `My Drive/xguard/` (`Xguard_defect.zip` has the labels, `defect.zip` the images).
2. Open the notebook in Colab, pick a T4 GPU runtime, set `ROOT` to that folder and run all cells. The model is saved to `ROOT/model/`.
3. Copy `detect.pt`, `classes.txt` and `model_info.json` into `models/` here and start `run_all.bat`.

`train/train.py` does the same thing locally if you have a GPU (`AUTO_TRAIN=1` in `.env`, or run it directly).

## Running each piece by hand

This is easier for debugging. Use a separate terminal for each.

| Terminal | Command | What you should see |
|---|---|---|
| 1 | `cd cse`, then `python -m acmecse --headless` | `CSE started` (web UI at http://127.0.0.1:8080/webui) |
| 2 | `python setup/setup_resources.py base` | all 2001 (4105 if they already exist) |
| 3 | `python gateway/gateway.py` | `① DET,1,0 -> ② arrivals A-00001` |
| 4 | `python gateway/image_source.py` | |
| 5 | `python ai/app.py` (or Docker, see below) | `모델 로드: detect.pt` |
| 6 | `streamlit run dashboard/app.py`, then open http://localhost:8501 once | this starts the :9100 receiver |
| 2 | `python setup/setup_resources.py subs` | six 2001s |

After that every product shows steps 1 through 9 in the logs. Resources and subscriptions stay
in the CSE database, so you don't need `base` or `subs` again after a restart. To start clean, run
`setup_resources.py clean` or `python -m acmecse --db-reset`.

Without hardware: with `SERIAL_PORT=SIM` (the default) a fake Arduino makes a product every 4
seconds. To test the flow without a model, set `MODEL_BACKEND=dummy`; any file whose name
contains a class name such as `knife` comes back as FAIL.

## Arduino

The pin-out is from the hardware team's code of 2026-11-15 (see `arduino/HARDWARE_NOTES_1115.txt`).

| Pin | Part | Role |
|---|---|---|
| D2 | E18-D80NK #1 (S1) | entry; sends `DET,seq,0` when a product comes in (internal pull-up) |
| D3 | E18-D80NK #2 (S2) | just before the rejector; handles the product at the head of the queue |
| D9 | MG996R servo | powered from the Arduino 5V/GND; only driven while pushing, then detached |
| D13 | onboard LED | on = running, blinking = stopped, off briefly while pushing |
| - | no e-stop button | use the Stop button on the dashboard (STOP command) |

Serial runs at 115200 baud. The Arduino sends `DET,seq,0`, `REJ,seq,DONE|TIMEOUT`, `PASS,seq` and
`STATUS,RUNNING|STOPPED`. The PC sends `OK,seq` / `NG,seq`, `STOP` / `START`, and for testing
`T1` (fake S1), `T2` (fake S2) and `SERVO` (one push).

Upload `arduino/` with PlatformIO, or open `arduino/xguard_2sensor/` in the Arduino IDE. In the
serial monitor, `T1` should give `DET,1,0`; then send `NG,1` and `T2` should give `REJ,1,DONE`.
Close the serial monitor and set `SERIAL_PORT` in `.env` to the right port, e.g. `COM4`. If the
gateway can't open the port it lists the ports it can see and retries every 3 seconds. The
dashboard's hardware test menu can also send T1, T2 and SERVO.

The trip from S1 to S2 has to take longer than the AI round trip (capture, upload, inference).
Otherwise the product reaches S2 with no verdict and is rejected as `TIMEOUT`. Servo angles and
timings are at the top of `main.cpp` and are tuned by the hardware team.

## Camera

`gateway/image_source.py` reads these from `.env`: `IMAGE_SOURCE=camera`, `CAMERA_INDEX=1`,
`CAMERA_BACKEND=MSMF` (DSHOW gives a black picture), `CAMERA_FOURCC=MJPG`, `CAMERA_ROTATE=180`
(the same orientation as the training photos), and `CAPTURE_DELAY_MS=0` when S1 is directly in
front of the camera.

About 3 seconds after start, `logs/image_source.log` shows a line starting with `카메라 확인: MSMF 1280x800`
with the measured brightness. With `SAVE_CAPTURES=1` every frame is also saved to `captures/`, which
is handy for collecting more training data.

## Dashboard

http://localhost:8501 shows the line state, counters (inspected, passed, rejected, rejected for
timeout, defect rate), the latest photo with defect boxes drawn on it, live rejectLog events,
the defect history found through discovery, and the hardware list. Counts start from when the
gateway was started.

## Running the AI in Docker or on MEC

```powershell
docker build -f ai/Dockerfile -t <dockerhub-id>/xguard-ai:1.0 .
docker run --rm -p 9000:9000 -e CSE_URL=http://host.docker.internal:8080 <dockerhub-id>/xguard-ai:1.0
docker push <dockerhub-id>/xguard-ai:1.0
```

On the MEC Sandbox VM, set `image` and `CSE_URL` (`http://<PC_IP>:8080`) in `mec/deployment.yaml`
and run `kubectl -n <SANDBOX_NS> apply -f mec/`. Then on the PC set `AI_NOTIFY_URL=http://<VM_IP>:30900/notify`
and `PC_IP=<the PC's real IP>` in `.env`, change `cseHost` in `cse/acme.ini` to the PC's IP, and run
`python setup/setup_resources.py base` again so the Inspection AE gets the new poa.

## Model file format

`ai/app.py` loads the model according to `MODEL_BACKEND` in `.env`.

### torchscript (default)

The file is `models/detect.pt`, a TorchScript model saved with
`torch.jit.save(torch.jit.trace(model.eval(), torch.zeros(1,3,640,640)), "detect.pt")`.
TorchScript lets us load it without the model's Python class. A file saved with
`torch.save(model)` won't work.

Input is a `float32` tensor `[1, 3, 640, 640]`, RGB, values 0 to 1 (pixel / 255). The image is
letterboxed: scaled to fit 640 with the aspect ratio kept, and the rest filled with gray
`(114,114,114)`. There's no mean/std normalization; if your model needs it, set `NORM_MEAN` and
`NORM_STD` in `.env`. `IMG_SIZE` changes the input size.

The output can be either of these:

- A (preferred): a float tensor `[N, 6]` or `[1, N, 6]` after NMS, one row per box, `[x1, y1, x2, y2, conf, cls]`.
  Coordinates are pixels in the 640x640 letterboxed input; `app.py` maps them back to the original
  image. `conf` is a 0-1 probability, `cls` is the class index (the line number in
  `models/classes.txt`, starting at 0). No boxes means an empty `[0, 6]` tensor.
- B: the raw YOLOv8 head, `[1, 4+num_classes, 8400]` (`cx, cy, w, h` plus a 0-1 score per class).
  This is what `yolo export format=torchscript` produces, and `app.py` runs NMS on it (IoU `IOU_TH`, 0.45).

If the model returns a tuple or list, the first element is used.

Class names go in `models/classes.txt`, one per line, a single lowercase English word each
(`knife`, `gun`, `scissors` and so on). This is the value that ends up in `defectClass`.

### ultralytics

To use an ultralytics `best.pt`, set `MODEL_PATH=models/best.pt` and `MODEL_BACKEND=ultralytics`,
and `pip install ultralytics`. Class names come from the model itself. If `MODEL_BACKEND` is
`torchscript` but the file isn't TorchScript, `app.py` falls back to ultralytics on its own. The
`detect.pt` made by `train/train.py` is format B TorchScript, so the defaults work.

### How the verdict is decided

If any box has `conf >= CONF_TH` (0.5) the verdict is FAIL, otherwise PASS. If the image can't
be read or inference fails, it's ERROR. `defectClass` is the class of the highest-scoring box and
`confidence` is its score, rounded to 3 places. On FAIL a cin also goes to `defects` with
`boxes: [[x1,y1,x2,y2,conf,"class"], ...]` in original image coordinates.

## Notes

- To register with an IN-CSE once one is chosen, fill in `registrarCse*` in `cse/acme.ini` and set `enableRemoteCSE=true` under `[cse]`.
- If flexContainers aren't available, set `RESULT_MODE=cin` in `.env`. inspectionResult becomes a normal container and everything else works the same.
- Subscription targets (`nu`) are AE-IDs such as `CRejector01`, so MQTT notifications arrive on `/oneM2M/req/id-mn/CRejector01/json`.
- `mosquitto_sub -t "/oneM2M/#" -v` shows the MQTT traffic, and `tests/requests.http` lets you try each step by hand.
