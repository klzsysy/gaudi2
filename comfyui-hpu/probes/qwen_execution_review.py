"""Collect natural-generation controls and render a four-pair review package."""
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def collect(root):
    import numpy as np
    from PIL import Image

    baseline_root = root.parent / "quality-qwen21-product-ab-2026-11-15"
    candidate_root = root / "workflow/gdn_direct-sdpa-whole"
    review = root / "review"
    rows = []
    sources = [(baseline_root / f"current-{index:02}.json",
                candidate_root / f"paired_seed_{index:02}-0.json") for index in range(1, 4)]
    sources.append((root / "workflow/reference/size512-0.json", candidate_root / "size512-0.json"))
    for index, source_pair in enumerate(sources, 1):
        name = f"{index:02}"
        arrays, captions, graphs, records = [], [], [], {}
        for label, source in zip(("reference", "candidate"), source_pair):
            history = json.loads(source.read_text())
            assert history["status"]["status_str"] == "success"
            assert history["prompt"][3].get("qwen_teacher") is None
            graph = history["prompt"][2]
            required = {key for key, node in graph.items()
                        if node["class_type"] in ("TextGenerate", "SamplerCustomAdvanced")}
            for kind, data in history["status"]["messages"]:
                if kind == "execution_cached":
                    assert not required.intersection(data["nodes"])
            normalized = copy.deepcopy(graph)
            normalized["461"]["inputs"].pop("filename_prefix")
            graphs.append(normalized)
            image = history["outputs"]["461"]["images"][0]
            original = Path("/home/sonny/ComfyUI-Gaudi/output") / image["subfolder"] / image["filename"]
            destination = review / label
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination / f"{name}.png")
            caption = history["outputs"]["459:472"]["text"][0]
            (destination / f"{name}.txt").write_text(caption)
            arrays.append(np.asarray(Image.open(original).convert("RGB")))
            captions.append(caption)
            records[label] = dict(history=str(source), prompt_id=history["prompt"][1], original=str(original),
                                  file_sha256=hashlib.sha256(original.read_bytes()).hexdigest())
        assert graphs[0] == graphs[1], name
        assert arrays[0].shape == arrays[1].shape
        difference = np.abs(arrays[0].astype(np.int16) - arrays[1].astype(np.int16))
        rows.append(dict(case=name, width=arrays[0].shape[1], height=arrays[0].shape[0],
                         text_seed=graphs[0]["459:471"]["inputs"]["sampling_mode.seed"],
                         image_seed=graphs[0]["459:488"]["inputs"]["noise_seed"],
                         caption_exact=captions[0] == captions[1], rgb_exact=bool(np.array_equal(*arrays)),
                         rgb_mae=float(difference.mean()), runs=records))
    manifest = dict(baseline="dfca7cd6", candidate="gdn_direct+sdpa+whole", pairs=rows,
                    prompt=graphs[0]["459:471"]["inputs"]["prompt"])
    (review / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def render(root):
    review = root / "review"
    manifest = json.loads((review / "manifest.json").read_text())
    pairs = review / "pairs"
    pairs.mkdir(exist_ok=True)
    thumbnails = []
    sections = []
    for row in manifest["pairs"]:
        name = row["case"]
        font_size = 52 if row["width"] > 1000 else 24
        command = ["magick"]
        for label, title in (("reference", "BASE dfca7cd6"), ("candidate", "TRIAL GDN+SDPA+graph")):
            command += ["(", str(review / label / f"{name}.png"), "-background", "#171c23",
                        "-fill", "white", "-font", "Arial", "-pointsize", str(font_size),
                        "-gravity", "north", "-splice", "0x96", "-annotate", "+0+18", f"{name}  {title}", ")"]
        subprocess.run(command + ["+append", str(pairs / f"{name}.png")], check=True)
        for column, label in enumerate(("reference", "candidate")):
            crop = f"{row['width']}x{row['height']}+{column * row['width']}+96"
            result = subprocess.run(["magick", "compare", "-metric", "AE", str(review / label / f"{name}.png"),
                                     f"{pairs / (name + '.png')}[{crop}]", "null:"], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
        if name != "04":
            thumbnail = pairs / f"{name}-preview.png"
            subprocess.run(["magick", str(pairs / f"{name}.png"), "-resize", "25%", str(thumbnail)], check=True)
            thumbnails.append(str(thumbnail))
        sections.append(f'''<section id="pair-{name}"><h2>对照 {name} · {row['width']} × {row['height']}</h2>
<p>文本 seed {row['text_seed']} · 图像 seed {row['image_seed']}</p>
<a href="pairs/{name}.png"><img src="pairs/{name}.png" alt="左侧已接受版本，右侧实验候选"></a>
<p><a href="reference/{name}.png">基线原图</a> · <a href="candidate/{name}.png">候选原图</a> ·
<a href="reference/{name}.txt">基线扩写</a> · <a href="candidate/{name}.txt">候选扩写</a></p></section>''')
    subprocess.run(["magick", "montage", *thumbnails, "-tile", "3x1", "-geometry", "+12+12",
                    "-background", "#f1f3f5", str(review / "overview.png")], check=True)
    for path in thumbnails:
        Path(path).unlink()
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen解码候选质量对照</title>
<style>body{{margin:24px auto;max-width:1600px;padding:0 20px;background:#f1f3f5;color:#171c23;font:16px/1.6 Inter,-apple-system,sans-serif}}img{{max-width:100%;height:auto}}section,details{{padding:20px;margin:24px 0;background:white;border-radius:8px}}a{{color:#1b4b86}}pre{{white-space:pre-wrap;font:inherit}}</style>
<h1>Qwen解码候选质量对照</h1>
<p>左侧为已接受产品 dfca7cd6，右侧为单token GDN直接状态更新、FP32 softmax SDPA和整步编译的实验组合。四组均自然扩写并完整出图，逐组使用相同提示词和两类seed。</p>
<p>01–03为1088×1920，04为512×512。点击对照图查看原尺寸，原PNG及扩写可分别打开。</p>
<img src="overview.png" alt="三组2MP自然出图对照概览">
<details><summary>原始提示词</summary><pre>{html.escape(manifest['prompt'])}</pre></details>
{''.join(sections)}</html>'''
    (review / "index.html").write_text(page)
    with zipfile.ZipFile(root / "qwen21-execution-quality.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(review.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(review))
    print(review / "index.html")


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--collect", action="store_true")
args = parser.parse_args()
collect(args.root) if args.collect else render(args.root)
