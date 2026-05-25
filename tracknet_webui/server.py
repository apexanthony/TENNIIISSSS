import cgi
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = APP_DIR / "uploads"
OUTPUT_DIR = APP_DIR / "outputs"
RUN_DIR = APP_DIR / "runs"

for directory in (UPLOAD_DIR, OUTPUT_DIR, RUN_DIR):
    directory.mkdir(parents=True, exist_ok=True)

JOBS = {}
JOBS_LOCK = threading.Lock()


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def safe_name(name):
    keep = []
    for char in Path(name or "upload.mp4").name:
        if char.isalnum() or char in ".-_":
            keep.append(char)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("._")
    return cleaned or "upload.mp4"


def is_inside(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def require_project_path(value, must_exist=True):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not is_inside(path, PROJECT_ROOT):
        raise ValueError("path outside project: {}".format(value))
    if must_exist and not path.exists():
        raise ValueError("path not found: {}".format(value))
    return path


def file_info(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "rel": str(path.relative_to(PROJECT_ROOT)) if is_inside(path, PROJECT_ROOT) else str(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def find_files(patterns, roots):
    files = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if ".git" in path.parts or "__pycache__" in path.parts:
                    continue
                resolved = path.resolve()
                if resolved in seen or not resolved.is_file():
                    continue
                seen.add(resolved)
                files.append(file_info(resolved))
    files.sort(key=lambda item: item["mtime"], reverse=True)
    return files


def json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_response(handler, message, status=400):
    json_response(handler, {"ok": False, "error": str(message)}, status=status)


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def str_arg(value, default):
    value = "" if value is None else str(value).strip()
    return value if value else str(default)


def int_arg(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def float_arg(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def bool_arg(value):
    return value is True or str(value).lower() in ("1", "true", "yes", "on")


def command_option_value(command, option, default=""):
    try:
        index = command.index(option)
    except ValueError:
        return default
    if index + 1 >= len(command):
        return default
    return str(command[index + 1]).strip().lower()


def command_uses_gpu(command):
    return command_option_value(command, "--device") == "cuda" or command_option_value(command, "--target") == "cuda"


def start_job(kind, command, cwd=PROJECT_ROOT, outputs=None, uses_gpu=None):
    if uses_gpu is None:
        uses_gpu = command_uses_gpu(command)
    job_id = "{}_{}_{}".format(kind, now_stamp(), uuid.uuid4().hex[:8])
    log_path = RUN_DIR / "{}.log".format(job_id)
    meta_path = RUN_DIR / "{}.json".format(job_id)
    with JOBS_LOCK:
        for existing in JOBS.values():
            refresh_job(existing)
        if uses_gpu:
            running_gpu = [
                existing
                for existing in JOBS.values()
                if existing.get("uses_gpu") and existing.get("returncode") is None
            ]
            if running_gpu:
                active = running_gpu[0]
                raise RuntimeError(
                    "已有 CUDA 任务正在运行：{}，请等待结束或先停止该任务".format(active["id"])
                )

        log_file = open(log_path, "wb")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        job = {
            "id": job_id,
            "kind": kind,
            "command": command,
            "cwd": str(cwd),
            "log_path": str(log_path),
            "outputs": outputs or [],
            "pid": process.pid,
            "started_at": time.time(),
            "returncode": None,
            "process": process,
            "log_file": log_file,
            "uses_gpu": uses_gpu,
        }
        JOBS[job_id] = job
    public_job = public_job_info(job)
    meta_path.write_text(json.dumps(public_job, ensure_ascii=False, indent=2), encoding="utf-8")
    return public_job


def refresh_job(job):
    process = job.get("process")
    if process is not None:
        rc = process.poll()
        if rc is not None and job.get("returncode") is None:
            job["returncode"] = rc
            job["ended_at"] = time.time()
            log_file = job.get("log_file")
            if log_file is not None:
                log_file.close()
                job["log_file"] = None
    return job


def public_job_info(job):
    refresh_job(job)
    return {
        "id": job["id"],
        "kind": job["kind"],
        "pid": job["pid"],
        "status": "running" if job.get("returncode") is None else ("success" if job["returncode"] == 0 else "failed"),
        "returncode": job.get("returncode"),
        "started_at": job.get("started_at"),
        "ended_at": job.get("ended_at"),
        "command": job["command"],
        "log_path": job["log_path"],
        "outputs": job.get("outputs", []),
        "uses_gpu": job.get("uses_gpu", False),
    }


def read_log_tail(path, max_bytes=60000):
    path = Path(path)
    if not path.exists():
        return ""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def config_payload():
    roots = [PROJECT_ROOT, UPLOAD_DIR, OUTPUT_DIR, PROJECT_ROOT / "exps"]
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "defaults": {
            "pt_model": str(PROJECT_ROOT / "exps" / "lite_heatmap_v3_270x480_from240" / "model_best_thr070_pw15.pt"),
            "onnx_model": str(PROJECT_ROOT / "exps" / "lite_heatmap_v3_270x480_from240" / "model_v3_270x480_b1_sigmoid.onnx"),
            "video": str(PROJECT_ROOT / "示例视频1.mp4"),
            "input_height": 270,
            "input_width": 480,
            "threshold": 0.70,
            "peak_window": 15,
        },
        "videos": find_files(["*.mp4", "*.webm", "*.avi", "*.mov", "*.mkv"], roots),
        "pt_models": find_files(["*.pt"], [PROJECT_ROOT / "exps", PROJECT_ROOT]),
        "onnx_models": find_files(["*.onnx"], [PROJECT_ROOT / "exps", PROJECT_ROOT]),
        "outputs": find_files(["*.webm", "*.mp4", "*.csv"], [OUTPUT_DIR, PROJECT_ROOT / "exps"]),
        "jobs": list_jobs(),
    }


def list_jobs():
    with JOBS_LOCK:
        return [public_job_info(job) for job in sorted(JOBS.values(), key=lambda item: item["started_at"], reverse=True)]


class TrackNetHandler(SimpleHTTPRequestHandler):
    server_version = "TrackNetWebUI/1.0"

    def log_message(self, fmt, *args):
        print("[webui] " + fmt % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.serve_static(STATIC_DIR / "index.html")
            elif parsed.path.startswith("/static/"):
                rel = unquote(parsed.path[len("/static/") :])
                self.serve_static(STATIC_DIR / rel)
            elif parsed.path == "/api/config":
                json_response(self, config_payload())
            elif parsed.path == "/api/jobs":
                json_response(self, {"ok": True, "jobs": list_jobs()})
            elif parsed.path.startswith("/api/jobs/"):
                self.handle_get_job(parsed.path)
            elif parsed.path == "/api/file":
                self.handle_file(parsed.query)
            elif parsed.path == "/api/preview":
                self.handle_preview(parsed.query)
            else:
                error_response(self, "not found", status=404)
        except Exception as exc:
            error_response(self, exc, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/upload":
                self.handle_upload()
            elif parsed.path == "/api/infer":
                self.handle_infer()
            elif parsed.path == "/api/eval":
                self.handle_eval()
            elif parsed.path == "/api/train":
                self.handle_train()
            elif parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/stop"):
                self.handle_stop_job(parsed.path)
            else:
                error_response(self, "not found", status=404)
        except Exception as exc:
            error_response(self, exc, status=500)

    def serve_static(self, path):
        path = Path(path).resolve()
        if not is_inside(path, STATIC_DIR) or not path.exists() or not path.is_file():
            error_response(self, "static file not found", status=404)
            return
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_file(self, query):
        params = parse_qs(query)
        value = params.get("path", [""])[0]
        path = require_project_path(value)
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        range_header = self.headers.get("Range", "")
        if range_header.startswith("bytes="):
            start_text, _, end_text = range_header[len("bytes=") :].partition("-")
            start = int(start_text or 0)
            end = int(end_text) if end_text else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */{}".format(size))
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def handle_preview(self, query):
        try:
            import cv2
        except ImportError as exc:
            error_response(self, "opencv is required for preview streaming: {}".format(exc), status=500)
            return

        params = parse_qs(query)
        value = params.get("path", [""])[0]
        path = require_project_path(value)
        max_width = int_arg(params.get("width", ["960"])[0], 960)

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            error_response(self, "cannot open preview video: {}".format(path), status=400)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / max(1.0, min(float(fps), 30.0))

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if max_width > 0 and width > max_width:
                    scale = max_width / float(width)
                    frame = cv2.resize(frame, (max_width, int(height * scale)))
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if not ok:
                    continue
                chunk = encoded.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write("Content-Length: {}\r\n\r\n".format(len(chunk)).encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            cap.release()

    def handle_get_job(self, path):
        job_id = path.rsplit("/", 1)[-1]
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            error_response(self, "job not found", status=404)
            return
        info = public_job_info(job)
        info["log"] = read_log_tail(info["log_path"])
        json_response(self, {"ok": True, "job": info})

    def handle_stop_job(self, path):
        job_id = path.split("/")[-2]
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if not job:
            error_response(self, "job not found", status=404)
            return
        refresh_job(job)
        if job.get("returncode") is None:
            job["process"].terminate()
            try:
                job["returncode"] = job["process"].wait(timeout=3)
            except subprocess.TimeoutExpired:
                job["process"].kill()
                job["returncode"] = job["process"].wait(timeout=3)
            job["ended_at"] = time.time()
            log_file = job.get("log_file")
            if log_file is not None:
                log_file.close()
                job["log_file"] = None
        json_response(self, {"ok": True, "job": public_job_info(job)})

    def handle_upload(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )
        item = form["video"] if "video" in form else None
        if item is None or not item.filename:
            error_response(self, "missing video file", status=400)
            return
        name = "{}_{}".format(now_stamp(), safe_name(item.filename))
        target = UPLOAD_DIR / name
        with open(target, "wb") as out:
            shutil.copyfileobj(item.file, out)
        json_response(self, {"ok": True, "file": file_info(target), "config": config_payload()})

    def handle_infer(self):
        data = read_json(self)
        engine = str_arg(data.get("engine"), "pt")
        video = require_project_path(data.get("video"))
        input_height = int_arg(data.get("input_height"), 270)
        input_width = int_arg(data.get("input_width"), 480)
        threshold = float_arg(data.get("threshold"), 0.70)
        peak_window = int_arg(data.get("peak_window"), 15)
        trace = int_arg(data.get("trace"), 7)
        stamp = now_stamp()
        stem = "{}_{}_{}x{}_thr{}_pw{}".format(
            stamp,
            engine,
            input_width,
            input_height,
            str(threshold).replace(".", ""),
            peak_window,
        )
        video_out = OUTPUT_DIR / "{}.mp4".format(stem)
        csv_out = OUTPUT_DIR / "{}.csv".format(stem)

        if engine == "pt":
            model = require_project_path(data.get("model"))
            command = [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "versions" / "v3_lightweight" / "infer_on_video_v3_batch.py"),
                "--model_path",
                str(model),
                "--video_path",
                str(video),
                "--video_out_path",
                str(video_out),
                "--csv_out_path",
                str(csv_out),
                "--base_channels",
                str(int_arg(data.get("base_channels"), 24)),
                "--input_height",
                str(input_height),
                "--input_width",
                str(input_width),
                "--batch_size",
                str(int_arg(data.get("batch_size"), 1)),
                "--threshold",
                str(threshold),
                "--peak_window",
                str(peak_window),
                "--trace",
                str(trace),
                "--codec",
                "mp4v",
                "--device",
                str_arg(data.get("device"), "cuda"),
                "--print_interval",
                "100",
            ]
        elif engine == "onnx":
            model = require_project_path(data.get("model"))
            command = [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "versions" / "v3_lightweight" / "infer_on_video_v3_onnx.py"),
                "--onnx_path",
                str(model),
                "--video_path",
                str(video),
                "--video_out_path",
                str(video_out),
                "--csv_out_path",
                str(csv_out),
                "--input_height",
                str(input_height),
                "--input_width",
                str(input_width),
                "--threshold",
                str(threshold),
                "--peak_window",
                str(peak_window),
                "--trace",
                str(trace),
                "--codec",
                "mp4v",
                "--target",
                str_arg(data.get("target"), "cpu"),
                "--print_interval",
                "100",
            ]
        else:
            raise ValueError("unknown inference engine: {}".format(engine))

        outputs = [file_info(video_out) if video_out.exists() else {"path": str(video_out), "name": video_out.name}]
        outputs.append(file_info(csv_out) if csv_out.exists() else {"path": str(csv_out), "name": csv_out.name})
        job = start_job("infer_" + engine, command, outputs=outputs)
        json_response(self, {"ok": True, "job": job})

    def handle_eval(self):
        data = read_json(self)
        model = require_project_path(data.get("model"))
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "versions" / "v3_lightweight" / "eval_thresholds_v3.py"),
            "--model-path",
            str(model),
            "--base-channels",
            str(int_arg(data.get("base_channels"), 24)),
            "--batch-size",
            str(int_arg(data.get("batch_size"), 4)),
            "--input-height",
            str(int_arg(data.get("input_height"), 270)),
            "--input-width",
            str(int_arg(data.get("input_width"), 480)),
            "--label-height",
            str(int_arg(data.get("label_height"), 720)),
            "--label-width",
            str(int_arg(data.get("label_width"), 1280)),
            "--heatmap-radius",
            str(int_arg(data.get("heatmap_radius"), 6)),
            "--heatmap-sigma",
            str(float_arg(data.get("heatmap_sigma"), 2.25)),
            "--device",
            str_arg(data.get("device"), "cuda"),
            "--thresholds",
            str_arg(data.get("thresholds"), "0.60,0.65,0.68,0.70,0.72,0.75,0.80,0.85,0.90"),
            "--min-dist",
            str(float_arg(data.get("min_dist"), 8.0)),
            "--peak-window",
            str(int_arg(data.get("peak_window"), 15)),
        ]
        job = start_job("eval", command)
        json_response(self, {"ok": True, "job": job})

    def handle_train(self):
        data = read_json(self)
        exp_id = str_arg(data.get("exp_id"), "webui_v3_run")
        command = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "versions" / "v3_lightweight" / "main_v3.py"),
            "--exp_id",
            exp_id,
            "--num_epochs",
            str(int_arg(data.get("num_epochs"), 20)),
            "--lr",
            str(float_arg(data.get("lr"), 2e-4)),
            "--batch_size",
            str(int_arg(data.get("batch_size"), 12)),
            "--steps_per_epoch",
            str(int_arg(data.get("steps_per_epoch"), 200)),
            "--val_intervals",
            str(int_arg(data.get("val_intervals"), 5)),
            "--base_channels",
            str(int_arg(data.get("base_channels"), 24)),
            "--input_height",
            str(int_arg(data.get("input_height"), 270)),
            "--input_width",
            str(int_arg(data.get("input_width"), 480)),
            "--label_height",
            str(int_arg(data.get("label_height"), 720)),
            "--label_width",
            str(int_arg(data.get("label_width"), 1280)),
            "--heatmap_radius",
            str(int_arg(data.get("heatmap_radius"), 6)),
            "--heatmap_sigma",
            str(float_arg(data.get("heatmap_sigma"), 2.25)),
            "--pos_weight",
            str(float_arg(data.get("pos_weight"), 120.0)),
            "--mse_weight",
            str(float_arg(data.get("mse_weight"), 1.0)),
            "--threshold",
            str(float_arg(data.get("threshold"), 0.70)),
            "--peak_window",
            str(int_arg(data.get("peak_window"), 15)),
            "--min_dist",
            str(float_arg(data.get("min_dist"), 8.0)),
            "--num_workers",
            str(int_arg(data.get("num_workers"), 0)),
            "--print_interval",
            str(int_arg(data.get("print_interval"), 20)),
            "--val_print_interval",
            str(int_arg(data.get("val_print_interval"), 50)),
            "--snapshot_interval",
            str(int_arg(data.get("snapshot_interval"), 25)),
            "--device",
            str_arg(data.get("device"), "cuda"),
        ]
        resume = str_arg(data.get("resume"), "")
        if resume:
            command.extend(["--resume", str(require_project_path(resume))])
        start_epoch = data.get("start_epoch")
        if start_epoch not in (None, ""):
            command.extend(["--start_epoch", str(int_arg(start_epoch, 0))])
        best_metric = data.get("best_metric")
        if best_metric not in (None, ""):
            command.extend(["--best_metric", str(float_arg(best_metric, 0.0))])
        if bool_arg(data.get("amp")):
            command.append("--amp")
        if bool_arg(data.get("augment")):
            command.append("--augment")
        outputs = [{"path": str(PROJECT_ROOT / "exps" / exp_id), "name": exp_id}]
        job = start_job("train", command, outputs=outputs)
        json_response(self, {"ok": True, "job": job})


def main():
    port = int(os.environ.get("TRACKNET_WEBUI_PORT", "8765"))
    server = ThreadingHTTPServer(("127.0.0.1", port), TrackNetHandler)
    print("TrackNet WebUI running at http://127.0.0.1:{}".format(port), flush=True)
    print("Project root: {}".format(PROJECT_ROOT), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
