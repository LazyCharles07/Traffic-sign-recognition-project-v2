"""本地两阶段路标推理：YOLOv8 检测框，手写 ResNet-18 识别含义。

把 Kaggle 笔记本导出的三个文件放在 weights/ 后启动：

    pip install -r requirements.txt
    python demo.py --yolo weights/best.pt --resnet weights/best_model.pth --data weights/dataset.yaml

浏览器打开 http://127.0.0.1:8000 ，上传街景图。服务只监听本机，不挂到局域网。
预处理 Resize(224) + ToTensor + ImageNet Normalize 只作用于裁剪出的路标小图。
检测输入边长保持训练时的 640，检测置信度下限 0.10。重叠达到 0.50 的框只留检测分更高的一个。分类裁剪相对检测框每边外扩 10%，画面上仍画原来的检测框。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.ops import nms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DET_CONF = 0.10
SAM_FLOOR = 0.05
SAM_LOW_MAX = 0.60
SAM_RELIABLE = 0.70
MIN_BOX = 16
YOLO_IMGSZ = 640
CROP_PAD = 0.10
NMS_IOU = 0.50
YOLO_PROBE_CONF = 0.001
SAM_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt"
MAX_MASK_AREA_RATIO = 8.0
BOX_PROMPT = (
    "找出图中每一块交通标志牌。不要框红绿灯、地面箭头、广告和普通路牌文字。"
    "只返回一个 JSON 数组，不要其他文字。"
    "每一项形如 {\"bbox\":[x1,y1,x2,y2]}。"
    "坐标是相对图片宽、高的千分比：左上角是 0，右下角是 1000，x 向右，y 向下。"
    "看不清的牌子不要写。"
)
PROVIDERS = (
    ("qwen", "QWEN", "千问", "qwen3.8-flash"),
    ("glm", "GLM", "智谱", "glm-5v-turbo"),
    ("deepseek", "DEEPSEEK", "DeepSeek", "deepseek-flash"),
)
SCHEME_NOTE = (
    "标准图标、需要准确位置时，用左边。牌子占满画面，或者画法和训练集不一样"
    "（例如白底红圈里写着「停」），左边容易漏框或认错。"
    "右边更适合描述它看到了什么，但给不出稳定的像素坐标，而且要联网、密钥，也更慢。"
)

# 与训练笔记本写入 dataset.yaml 的规则一致。推理优先使用 yaml / checkpoint 里的含义。
MEANING_TABLE = {
    "i1": "步行",
    "i2": "非机动车行驶",
    "i3": "环岛行驶",
    "i4": "机动车行驶",
    "i5": "靠右侧道路行驶",
    "i6": "靠左侧道路行驶",
    "i7": "立体交叉直行和右转弯行驶",
    "i8": "立体交叉直行和左转弯行驶",
    "i9": "鸣喇叭",
    "i10": "向右转弯",
    "i11": "向左向右转弯",
    "i12": "向左转弯",
    "i13": "直行",
    "i14": "直行和向右转弯",
    "i15": "直行和向左转弯",
    "il50": "最低限速50",
    "il60": "最低限速60",
    "il70": "最低限速70",
    "il80": "最低限速80",
    "il90": "最低限速90",
    "il100": "最低限速100",
    "il110": "最低限速110",
    "io": "其他指示标志",
    "ip": "人行横道",
    "p1": "禁止超车",
    "p2": "禁止畜力车进入",
    "p3": "禁止大型客车驶入",
    "p4": "禁止电动三轮车驶入",
    "p5": "禁止掉头",
    "p6": "禁止非机动车进入",
    "p7": "禁止载货汽车左转",
    "p8": "禁止汽车拖、挂车驶入",
    "p9": "禁止行人进入",
    "p10": "禁止机动车驶入",
    "p11": "禁止鸣喇叭",
    "p12": "禁止二轮摩托车驶入",
    "p13": "禁止某两种车驶入",
    "p14": "禁止直行",
    "p15": "禁止人力车进入",
    "p16": "禁止人力货运三轮车进入",
    "p17": "禁止人力客运三轮车进入",
    "p18": "禁止三轮车机动车通行",
    "p19": "禁止向右转弯",
    "p20": "禁止向左向右转弯",
    "p21": "禁止直行和向右转弯",
    "p23": "禁止向左转弯",
    "p26": "禁止载货汽车驶入",
    "p27": "禁止运输危险物品车辆驶入",
    "p28": "禁止直行和向左转弯",
    "pb": "禁止通行",
    "pbp": "禁止自行车和行人通行",
    "pg": "减速让行",
    "ph": "限高",
    "pn": "禁止停车",
    "pne": "禁止驶入",
    "po": "其他禁止标志",
    "pa": "限制轴重",
    "pm": "限重",
    "pr": "解除限速",
    "pr40": "解除限速40",
    "w": "警告标志",
    "ps": "停车让行",
    "w13": "十字交叉",
    "w32": "施工",
    "w55": "注意儿童",
    "w57": "注意行人",
    "w59": "注意合流",
    "wo": "其他警告标志",
}

CROP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>交通路标识别</title>
  <style>
    :root {
      --ink: #142018;
      --paper: #e4eee6;
      --card: #f7f3ea;
      --ring: #9d1c1c;
      --yellow: #e2b100;
      --muted: #3e5146;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    }
    header { padding: 28px 32px 0; max-width: 1280px; margin: 0 auto; }
    h1 {
      margin: 0 0 8px;
      font-family: "Songti SC", "STSong", "SimSun", serif;
      font-size: 34px;
      font-weight: 600;
      letter-spacing: 0.04em;
    }
    .lead { margin: 0 0 16px; color: var(--muted); max-width: 46em; }
    .hazard { height: 12px; background: repeating-linear-gradient(-45deg, var(--yellow) 0 14px, #1a1a1a 14px 28px); }
    main { max-width: 1280px; margin: 0 auto; padding: 22px 32px 48px; }
    form { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
    input[type="file"] { color: var(--ink); }
    button {
      background: var(--ring);
      color: white;
      border: 0;
      padding: 8px 16px;
      font: inherit;
      cursor: pointer;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    button:focus-visible, input:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
    #chosen { display: block; max-width: 280px; max-height: 160px; object-fit: contain; background: white; border: 1px solid #c9d5cc; margin-bottom: 16px; }
    .status { min-height: 1.4em; margin: 0 0 14px; }
    .err { color: var(--ring); }
    .board { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .panel { background: var(--card); border-top: 4px solid var(--ring); padding: 14px; }
    .panel h2 { margin: 0 0 10px; font-size: 18px; font-family: "Songti SC", "STSong", "SimSun", serif; }
    .panel img { width: 100%; height: auto; background: white; display: block; }
    .panel ul { margin: 10px 0 0; padding-left: 18px; }
    .panel li { margin: 4px 0; }
    .empty { color: var(--muted); margin: 8px 0 0; }
    @media (max-width: 860px) {
      header, main { padding-left: 16px; padding-right: 16px; }
      .board { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto; } }
  </style>
</head>
<body>
  <header>
    <h1>路标对照</h1>
    <p class="lead">上传一张街景。下面四张图用的是同一张照片：第一张由检测模型找框并补全，后三张由千问、智谱、DeepSeek 找框。含义都由分类模型认。</p>
  </header>
  <div class="hazard" aria-hidden="true"></div>
  <main>
    <form id="form">
      <input id="file" type="file" name="file" accept="image/*" required>
      <button id="submit" type="submit">识别</button>
    </form>
    <img id="chosen" alt="刚选择的图片" hidden>
    <p id="message" class="status"></p>
    <div class="board" id="board"></div>
  </main>
  <script>
    const panels = [
      { id: "yolo", title: "检测" },
      { id: "qwen", title: "千问" },
      { id: "glm", title: "智谱" },
      { id: "deepseek", title: "DeepSeek" }
    ];
    const board = document.getElementById("board");
    const message = document.getElementById("message");
    const chosen = document.getElementById("chosen");
    const fileInput = document.getElementById("file");
    const submitButton = document.getElementById("submit");
    let serial = 0;
    let chosenUrl = "";

    function esc(text) {
      return String(text).replace(/[&<>"]/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"
      }[ch]));
    }
    function renderShell() {
      board.innerHTML = panels.map((panel) => `
        <section class="panel" data-id="${panel.id}">
          <h2>${esc(panel.title)}</h2>
          <img alt="" hidden>
          <p class="empty">还没有结果</p>
          <ul hidden></ul>
        </section>
      `).join("");
    }
    function clearPanels() {
      document.querySelectorAll(".panel").forEach((panel) => {
        const img = panel.querySelector("img");
        img.removeAttribute("src");
        img.hidden = true;
        panel.querySelector("ul").hidden = true;
        panel.querySelector("ul").innerHTML = "";
        panel.querySelector(".empty").hidden = false;
        panel.querySelector(".empty").textContent = "还没有结果";
        panel.querySelector(".empty").classList.remove("err");
      });
    }
    function showWaiting() {
      document.querySelectorAll(".panel .empty").forEach((node) => {
        node.hidden = false;
        node.classList.remove("err");
        node.textContent = "正在识别…";
      });
    }
    fileInput.addEventListener("change", () => {
      serial += 1;
      clearPanels();
      message.className = "status";
      const file = fileInput.files[0];
      if (chosenUrl) URL.revokeObjectURL(chosenUrl);
      if (!file) {
        chosen.hidden = true;
        chosen.removeAttribute("src");
        message.textContent = "";
        return;
      }
      chosenUrl = URL.createObjectURL(file);
      chosen.src = chosenUrl;
      chosen.hidden = false;
      message.textContent = "已换上这张图。按识别后，四张结果会一起更新。";
    });
    document.getElementById("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const mySerial = ++serial;
      clearPanels();
      showWaiting();
      message.className = "status";
      message.textContent = "四张图都在识别…";
      submitButton.disabled = true;
      try {
        const response = await fetch("/predict", { method: "POST", body: new FormData(event.target) });
        const data = await response.json();
        if (mySerial !== serial) return;
        if (!response.ok) throw new Error(data.detail || "识别失败");
        (data.panels || []).forEach((panel) => {
          const node = board.querySelector(`[data-id="${panel.id}"]`);
          if (!node) return;
          const img = node.querySelector("img");
          const list = node.querySelector("ul");
          const empty = node.querySelector(".empty");
          if (panel.error) {
            empty.hidden = false;
            empty.classList.add("err");
            empty.textContent = panel.error;
            return;
          }
          img.src = "";
          img.src = "data:image/jpeg;base64," + panel.image_base64;
          img.hidden = false;
          list.innerHTML = (panel.signs || []).map((sign) => {
            const det = sign.det_conf == null ? "" : `，检测 ${Number(sign.det_conf).toFixed(2)}`;
            return `<li>${esc(sign.meaning)} ${Number(sign.cls_conf).toFixed(2)}${det}</li>`;
          }).join("");
          list.hidden = !(panel.signs || []).length;
          empty.hidden = true;
          if (!(panel.signs || []).length) {
            empty.hidden = false;
            empty.classList.remove("err");
            empty.textContent = "没有框出标志";
          }
        });
        if (mySerial === serial) message.textContent = "这一张的四份结果已更新。";
      } catch (error) {
        if (mySerial !== serial) return;
        message.className = "status err";
        message.textContent = error.message;
        document.querySelectorAll(".panel .empty").forEach((node) => {
          if (node.textContent === "正在识别…") node.textContent = "这次没有结果";
        });
      } finally {
        if (mySerial === serial) submitButton.disabled = false;
      }
    });
    renderShell();
  </script>
</body>
</html>
"""


class BasicBlock(nn.Module):
    """ResNet 基础残差块。通道或步长变化时，用 1x1 卷积把捷径分支对齐到主分支。"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=block_stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class UnicodeJSONResponse(JSONResponse):
    def render(self, content):
        return json.dumps(content, ensure_ascii=False).encode("utf-8")


def meaning_of(code):
    code = str(code)
    if code in MEANING_TABLE:
        return MEANING_TABLE[code]
    matched = re.fullmatch(r"pl(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限速{matched.group(1)}"
    matched = re.fullmatch(r"il(\d+(?:\.\d+)?)", code)
    if matched:
        return f"最低限速{matched.group(1)}"
    matched = re.fullmatch(r"ph(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限高{matched.group(1)}米"
    matched = re.fullmatch(r"pm(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限重{matched.group(1)}吨"
    matched = re.fullmatch(r"pw(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限宽{matched.group(1)}米"
    matched = re.fullmatch(r"pr(\d+(?:\.\d+)?)", code)
    if matched:
        return f"解除限速{matched.group(1)}"
    matched = re.fullmatch(r"pa(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限制轴重{matched.group(1)}吨"
    return code


def as_string_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        ordered = sorted((int(key), str(item)) for key, item in value.items())
        return [item for _, item in ordered]
    raise TypeError(f"无法解析名单: {type(value)}")


def load_dataset_spec(path):
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"{path} 缺少 names")
    class_names = as_string_list(data["names"])
    if "meanings" in data:
        meanings = as_string_list(data["meanings"])
        if len(meanings) != len(class_names):
            raise ValueError("dataset.yaml 中 meanings 与 names 长度不一致")
    else:
        meanings = [meaning_of(name) for name in class_names]
    return class_names, meanings


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        while name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    return cleaned


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_font(size=18):
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def clip_box(bbox, width, height, min_side):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    if min(x2 - x1, y2 - y1) < min_side:
        return None
    return x1, y1, x2, y2


def expand_box(bbox, width, height, pad_ratio):
    """分类裁剪每边按框宽、框高外扩。画框不用这个结果。"""
    x1, y1, x2, y2 = bbox
    if pad_ratio <= 0:
        return x1, y1, x2, y2
    dx = (x2 - x1) * pad_ratio
    dy = (y2 - y1) * pad_ratio
    ex1 = max(0, int(x1 - dx))
    ey1 = max(0, int(y1 - dy))
    ex2 = min(width, int(round(x2 + dx)))
    ey2 = min(height, int(round(y2 + dy)))
    if ex2 <= ex1 or ey2 <= ey1:
        return x1, y1, x2, y2
    return ex1, ey1, ex2, ey2


def sam_weight_path():
    return Path(__file__).resolve().parent / "weights" / "mobile_sam.pt"


def ensure_sam_weight():
    path = sam_weight_path()
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SAM_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def mask_to_box(mask, image_width, image_height):
    if mask.ndim != 2:
        return None
    ys, xs = torch.where(mask)
    if xs.numel() == 0:
        return None
    mask_height, mask_width = mask.shape
    return [
        float(xs.min().item()) * image_width / mask_width,
        float(ys.min().item()) * image_height / mask_height,
        float(xs.max().item() + 1) * image_width / mask_width,
        float(ys.max().item() + 1) * image_height / mask_height,
    ]


def sam_result_reliable(original, refined):
    """补完后的分类把握本身比较高就采用。原来已经很有把握、新框反而缩小的，仍用原框。"""
    if refined["cls_conf"] < SAM_RELIABLE:
        return False
    ox1, oy1, ox2, oy2 = original["bbox"]
    rx1, ry1, rx2, ry2 = refined["bbox"]
    old_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
    new_area = max(0.0, (rx2 - rx1) * (ry2 - ry1))
    if original["cls_conf"] >= 0.90 and new_area < old_area:
        return False
    return True


def accept_refined_box(original, refined, center):
    """补全后的框必须仍包住提示点，并且不能大到把整张图都圈进来。"""
    ox1, oy1, ox2, oy2 = original
    rx1, ry1, rx2, ry2 = refined
    cx, cy = center
    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
        return False
    old_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
    new_area = max(0.0, (rx2 - rx1) * (ry2 - ry1))
    return 0 < new_area <= old_area * MAX_MASK_AREA_RATIO


def suppress_overlaps(boxes, iou_threshold):
    """按位置去重，不看检测类别。重叠达到阈值时只留检测分更高的框。"""
    if len(boxes) <= 1:
        return boxes
    coords = torch.tensor([item["bbox"] for item in boxes], dtype=torch.float32)
    scores = torch.tensor([item["det_conf"] for item in boxes], dtype=torch.float32)
    keep = nms(coords, scores, iou_threshold)
    return [boxes[int(index)] for index in keep.tolist()]


def preprocess_crop(crop):
    """只预处理路标小图，不处理整张街景。"""
    return CROP_TRANSFORM(crop)


class Predictor:
    def __init__(
        self,
        yolo_path,
        resnet_path,
        data_path,
        det_conf=DET_CONF,
        min_box=MIN_BOX,
        imgsz=YOLO_IMGSZ,
        crop_pad=CROP_PAD,
    ):
        self.det_conf = det_conf
        self.min_box = min_box
        self.imgsz = imgsz
        self.crop_pad = crop_pad
        self.device = select_device()
        self.font = load_font()
        self.lock = threading.Lock()
        yaml_names, yaml_meanings = load_dataset_spec(data_path)
        checkpoint = torch_load(resnet_path, self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            checkpoint_names = checkpoint.get("class_names")
            checkpoint_meanings = checkpoint.get("meanings")
            normalize = checkpoint.get("normalize")
        else:
            state_dict = checkpoint
            checkpoint_names = None
            checkpoint_meanings = None
            normalize = None
        if normalize is not None:
            mean = tuple(normalize["mean"])
            std = tuple(normalize["std"])
            if mean != IMAGENET_MEAN or std != IMAGENET_STD:
                raise RuntimeError("checkpoint 的 Normalize 与本地预处理不一致")
        if checkpoint_names is not None and [str(name) for name in checkpoint_names] != yaml_names:
            raise RuntimeError(
                "dataset.yaml 的 names 与 best_model.pth 的 class_names 不一致。"
                "请使用同一次训练导出的 best.pt、best_model.pth 和 dataset.yaml。"
            )
        self.class_names = yaml_names
        if checkpoint_meanings is not None and [str(item) for item in checkpoint_meanings] != yaml_meanings:
            print("警告: checkpoint 与 dataset.yaml 的含义不一致，界面使用 dataset.yaml。")
        self.meanings = yaml_meanings
        state_dict = strip_module_prefix(state_dict)
        self.resnet = ResNet18(num_classes=len(self.class_names)).to(self.device)
        self.resnet.load_state_dict(state_dict)
        self.resnet.eval()
        weight = self.resnet.fc.weight
        if weight.shape[0] != len(self.class_names):
            raise RuntimeError("ResNet 输出维度与类别数不一致")
        from ultralytics import YOLO

        self.yolo = YOLO(str(yolo_path))
        self.sam = None
        try:
            from ultralytics import SAM

            self.sam = SAM(str(ensure_sam_weight()))
            print("SAM: 已加载，用框中心点把半截框补成整块牌子")
        except Exception as exc:
            print(f"SAM 未加载，继续使用 YOLO 框: {exc}")
        print(f"设备: {self.device}")
        print(f"类别数: {len(self.class_names)}")
        print(f"检测输入边长: {self.imgsz}")
        print(f"检测置信度下限: {self.det_conf}")
        print(f"补框只处理检测把握低于 {SAM_LOW_MAX:.2f} 的框，补完后分类把握达到 {SAM_RELIABLE:.2f} 才采用")
        print(f"最终筛查: 分类把握 >= {SAM_RELIABLE:.2f}")
        print(f"分类裁剪外扩: {self.crop_pad:.0%}")
        print(f"重叠去重 IoU: {NMS_IOU:.2f}")

    def detect_boxes(self, image):
        device_arg = 0 if self.device.type == "cuda" else "cpu"
        results = self.yolo.predict(
            source=image,
            conf=YOLO_PROBE_CONF,
            imgsz=self.imgsz,
            verbose=False,
            device=device_arg,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        boxes = []
        for box, score in zip(xyxy, scores):
            boxes.append({"bbox": [float(v) for v in box.tolist()], "det_conf": float(score)})
        return boxes

    def refine_boxes(self, image, boxes):
        """用每个框的中心点提示 SAM。掩码盖住整块牌子后，换成它的外接矩形。"""
        if self.sam is None or not boxes:
            return boxes
        width, height = image.size
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
        device_arg = 0 if self.device.type == "cuda" else "cpu"
        try:
            results = self.sam.predict(
                source=image,
                points=centers,
                labels=[1] * len(centers),
                verbose=False,
                device=device_arg,
            )
        except Exception as exc:
            print(f"SAM 补框失败，继续使用 YOLO 框: {exc}")
            return boxes
        masks = results[0].masks
        if masks is None or masks.data is None or masks.data.shape[0] != len(boxes):
            return boxes
        data = masks.data.detach().cpu()
        refined = []
        for box, center, mask in zip(boxes, centers, data):
            candidate = mask_to_box(mask > 0.5, width, height)
            if candidate is not None and accept_refined_box(box["bbox"], candidate, center):
                refined.append({**box, "bbox": candidate})
            else:
                refined.append(box)
        return refined

    def classify_boxes(self, image, boxes):
        width, height = image.size
        prepared = []
        for box in boxes:
            clipped = clip_box(box["bbox"], width, height, self.min_box)
            if clipped is None:
                prepared.append(None)
                continue
            expanded = expand_box(clipped, width, height, self.crop_pad)
            prepared.append({**box, "bbox": list(clipped), "tile": image.crop(expanded)})
        valid = [item for item in prepared if item is not None]
        if not valid:
            return prepared
        batch = torch.stack([preprocess_crop(item["tile"]) for item in valid], dim=0)
        batch = batch.to(self.device)
        with torch.no_grad():
            logits = self.resnet(batch)
            probabilities = torch.softmax(logits, dim=1)
            confidences, class_ids = probabilities.max(dim=1)
        for item, class_id, confidence in zip(valid, class_ids.tolist(), confidences.tolist()):
            item["class_id"] = int(class_id)
            item["cls_conf"] = float(confidence)
            item["code"] = self.class_names[item["class_id"]]
            item["meaning"] = self.meanings[item["class_id"]]
        return prepared

    def predict(self, image):
        # 1. 接收任意尺寸的原始图片。后续几何操作都回到这张图的坐标系。
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL.Image")
        image = image.convert("RGB")
        width, height = image.size

        # 2. 把整张原图送入 YOLO，得到边界框和检测置信度。
        detected = self.detect_boxes(image)

        # 3. 放宽到 0.05 才进入候选。高把握的框不补。低把握的框才试 SAM。
        detected = [box for box in detected if box["det_conf"] >= SAM_FLOOR]
        detected = suppress_overlaps(detected, NMS_IOU)
        steady = [box for box in detected if box["det_conf"] >= SAM_LOW_MAX]
        tentative = [box for box in detected if box["det_conf"] < SAM_LOW_MAX]
        crops = [item for item in self.classify_boxes(image, steady) if item is not None]
        original_low = self.classify_boxes(image, tentative)
        refined_low = self.classify_boxes(image, self.refine_boxes(image, tentative))
        for original, refined, box in zip(original_low, refined_low, tentative):
            if original is None:
                continue
            if refined is not None and sam_result_reliable(original, refined):
                crops.append(refined)
            else:
                crops.append(original)

        # 宽进严出：0.05 的框可以进来试补。最后只去掉分类把握低于 0.70 的结果。
        crops = [item for item in crops if item is not None and item["cls_conf"] >= SAM_RELIABLE]

        # 4. 分类已完成。画面上只留下通过最终筛查的框。
        return self.draw_signs(image, crops)

    def draw_signs(self, image, crops):
        width, height = image.size
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        palette = [(220, 68, 55), (25, 118, 210), (46, 125, 50), (245, 124, 0), (123, 31, 162), (0, 121, 107)]
        signs = []
        for index, item in enumerate(crops):
            color = palette[index % len(palette)]
            x1, y1, x2, y2 = item["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{item['meaning']} {item['cls_conf']:.2f}"
            text_box = draw.textbbox((0, 0), text, font=self.font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            text_y = y1 - text_h - 6
            if text_y < 0:
                text_y = min(y1 + 2, max(0, height - text_h - 4))
            draw.rectangle([x1, text_y, min(width, x1 + text_w + 8), text_y + text_h + 4], fill=color)
            draw.text((x1 + 4, text_y + 2), text, fill=(255, 255, 255), font=self.font)
            det_conf = item.get("det_conf")
            signs.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": item["class_id"],
                    "code": item["code"],
                    "meaning": item["meaning"],
                    "det_conf": None if det_conf is None else round(float(det_conf), 4),
                    "cls_conf": round(item["cls_conf"], 4),
                }
            )
        return annotated, signs


def load_dotenv(path):
    file_path = Path(path)
    if not file_path.is_file():
        return
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_providers():
    providers = []
    for provider_id, prefix, title, default_model in PROVIDERS:
        base = os.environ.get(f"{prefix}_API_BASE", "").strip().rstrip("/")
        key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        model = os.environ.get(f"{prefix}_MODEL", "").strip() or default_model
        if base and not base.endswith("/chat/completions"):
            base = base + "/chat/completions"
        providers.append(
            {
                "id": provider_id,
                "title": title,
                "base": base,
                "key": key,
                "model": model,
                "configured": bool(base and key and model),
            }
        )
    return providers


def vision_image_jpeg(image):
    """发给大模型的是同一张图的压缩副本，避免请求过大。"""
    copy = image.copy()
    long_side = max(copy.size)
    if long_side > 1600:
        scale = 1600 / long_side
        copy = copy.resize((max(1, int(copy.width * scale)), max(1, int(copy.height * scale))))
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def vision_message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in (None, "text"):
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return ""


def parse_model_boxes(text, width, height):
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match is None:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("boxes") or payload.get("signs") or []
    boxes = []
    if not isinstance(payload, list):
        return boxes
    for item in payload:
        if isinstance(item, dict):
            raw = next((item[key] for key in ("bbox", "bbox_2d", "box", "box_2d") if key in item), None)
        else:
            raw = item
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in raw]
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        largest = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if largest <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        elif largest <= 1000:
            x1, x2 = x1 / 1000 * width, x2 / 1000 * width
            y1, y2 = y1 / 1000 * height, y2 / 1000 * height
        boxes.append({"bbox": [x1, y1, x2, y2], "det_conf": None})
        if len(boxes) >= 12:
            break
    return boxes


def request_model_boxes(provider, image):
    if not provider["configured"]:
        return [], f"{provider['title']}未配置。在 .env 里填写 {provider['id'].upper()} 的地址、密钥和模型名后重启。"
    body = {
        "model": provider["model"],
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BOX_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + vision_image_jpeg(image)}},
                ],
            }
        ],
    }
    if "glm" in provider["model"].lower():
        body["thinking"] = {"type": "disabled"}
    request = urllib.request.Request(
        provider["base"],
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + provider["key"], "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return [], f"{provider['title']}请求失败：{exc.code} {detail}"
    except Exception as exc:
        return [], f"{provider['title']}请求失败：{exc}"
    choices = payload.get("choices") or []
    if not choices:
        return [], f"{provider['title']}没有返回内容"
    text = vision_message_text(choices[0].get("message") or {})
    width, height = image.size
    return parse_model_boxes(text, width, height), ""


def encode_jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


app = FastAPI(title="交通路标识别")
STATE = {"predictor": None, "args": None}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


def panel_from_crops(predictor, image, panel_id, title, crops, error=""):
    if error:
        blank, _signs = predictor.draw_signs(image, [])
        return {"id": panel_id, "title": title, "image_base64": encode_jpeg(blank), "signs": [], "error": error}
    annotated, signs = predictor.draw_signs(image, crops)
    return {"id": panel_id, "title": title, "image_base64": encode_jpeg(annotated), "signs": signs, "error": ""}


@app.post("/predict", response_class=UnicodeJSONResponse)
async def predict_api(file: UploadFile = File(...)):
    predictor = STATE["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
        image = image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法解析为图片") from exc
    providers = load_providers()
    copies = [image.copy() for _provider in providers]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(request_model_boxes, provider, copy)
            for provider, copy in zip(providers, copies)
        ]
        with predictor.lock:
            annotated, signs = predictor.predict(image)
        fetched = [future.result() for future in futures]
        panels = [
            {
                "id": "yolo",
                "title": "检测",
                "image_base64": encode_jpeg(annotated),
                "signs": signs,
                "error": "",
            }
        ]
        with predictor.lock:
            for provider, (boxes, error) in zip(providers, fetched):
                if error:
                    panels.append(panel_from_crops(predictor, image, provider["id"], provider["title"], [], error))
                    continue
                crops = [
                    item
                    for item in predictor.classify_boxes(image, boxes)
                    if item is not None and item["cls_conf"] >= SAM_RELIABLE
                ]
                panels.append(panel_from_crops(predictor, image, provider["id"], provider["title"], crops))
    return {"panels": panels}


def parse_args():
    parser = argparse.ArgumentParser(description="本地交通路标识别服务")
    parser.add_argument("--yolo", default="weights/best.pt", help="YOLOv8 best.pt")
    parser.add_argument("--resnet", default="weights/best_model.pth", help="ResNet-18 best_model.pth")
    parser.add_argument("--data", default="weights/dataset.yaml", help="训练导出的 dataset.yaml")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--det-conf", type=float, default=DET_CONF)
    parser.add_argument("--min-box", type=int, default=MIN_BOX)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument("--crop-pad", type=float, default=CROP_PAD)
    return parser.parse_args()


def require_weight(path):
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"找不到 {file_path}。请把 Kaggle Output 中的 best.pt、best_model.pth、dataset.yaml 放到 weights/。"
        )
    return file_path


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()
    yolo_path = require_weight(args.yolo)
    resnet_path = require_weight(args.resnet)
    data_path = require_weight(args.data)
    STATE["args"] = args
    STATE["predictor"] = Predictor(
        yolo_path,
        resnet_path,
        data_path,
        det_conf=args.det_conf,
        min_box=args.min_box,
        imgsz=args.imgsz,
        crop_pad=args.crop_pad,
    )
    print(f"只在本机提供服务: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
