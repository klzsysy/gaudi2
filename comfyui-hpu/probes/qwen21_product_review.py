"""Package original PNGs and deterministic side-by-side comparison sheets."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import zipfile


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
review = args.root / "review"
manifest = json.loads((review / "manifest.json").read_text())
pairs = review / "pairs"
pairs.mkdir(exist_ok=True)
thumbnails = []
for row in manifest["pairs"]:
    name = row["case"]
    command = ["magick"]
    for version, title in (("current", "CURRENT"), ("previous", "PREVIOUS")):
        commit = manifest["versions"][version]["commit"][:8]
        command += ["(", str(review / version / f"{name}.png"), "-background", "#171c23",
                    "-fill", "white", "-font", "Arial", "-pointsize", "52", "-gravity", "north",
                    "-splice", "0x96", "-annotate", "+0+18", f"{name}   {title} {commit}", ")"]
    command += ["+append", str(pairs / f"{name}.png")]
    subprocess.run(command, check=True)
    # Verify that each full-resolution panel contains the original image pixels.
    for column, version in enumerate(("current", "previous")):
        crop = f"{row['width']}x{row['height']}+{column * row['width']}+96"
        result = subprocess.run(["magick", "compare", "-metric", "AE", str(review / version / f"{name}.png"),
                                 f"{pairs / (name + '.png')}[{crop}]", "null:"], capture_output=True, text=True)
        assert result.returncode == 0, (name, version, result.stderr)
    thumbnail = pairs / f"{name}-preview.png"
    subprocess.run(["magick", str(pairs / f"{name}.png"), "-resize", "25%", str(thumbnail)], check=True)
    thumbnails.append(str(thumbnail))
subprocess.run(["magick", "montage", *thumbnails, "-tile", "4x2", "-geometry", "+12+12",
                "-background", "#f1f3f5", str(review / "overview.png")], check=True)
for path in thumbnails:
    Path(path).unlink()

navigation = " ".join(f'<a href="#case-{row["case"]}">{row["case"]}</a>' for row in manifest["pairs"])
sections = []
for row in manifest["pairs"]:
    name = row["case"]
    sections.append(f'''<section id="case-{name}">
      <h2>对照 {name}</h2>
      <p>文本 seed {row['text_seed']} · 出图 seed {row['image_seed']}</p>
      <a href="pairs/{name}.png" target="_blank"><img src="pairs/{name}.png" alt="第{name}组，左侧当前版本，右侧上一版本" loading="lazy" width="{row['width']*2}" height="{row['height']+96}"></a>
      <div class="links"><a href="current/{name}.png" target="_blank">当前版本原图</a><a href="previous/{name}.png" target="_blank">上一版本原图</a><a href="current/{name}.txt" target="_blank">当前扩写</a><a href="previous/{name}.txt" target="_blank">上一版扩写</a></div>
    </section>''')
prompt = html.escape(manifest["prompt"].replace("\\n", "\n"))
page = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen Image 2.1 · 8组版本对照</title>
<style>
* {{ box-sizing:border-box }}
body {{ margin:0;background:#f1f3f5;color:#171c23;font:16px/1.6 Inter,-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif }}
main {{ max-width:1500px;margin:auto;padding:28px }}
h1 {{ font-size:26px;margin:0 0 8px }}
h2 {{ font-size:22px;margin:0 }}
p {{ margin:8px 0 16px;color:#46505d }}
nav {{ position:sticky;top:0;z-index:1;padding:12px 0;background:#f1f3f5 }}
nav a {{ display:inline-block;padding:6px 16px;margin:0 4px 4px 0;background:white;border:1px solid #cdd4dd;border-radius:6px }}
a {{ color:#1b4b86;text-decoration:none }}
a:hover {{ text-decoration:underline }}
img {{ display:block;max-width:100%;height:auto }}
section {{ background:white;padding:20px;margin:24px 0;border:1px solid #dce1e7;border-radius:8px;scroll-margin-top:75px }}
.links {{ display:flex;flex-wrap:wrap;gap:12px 28px;padding-top:16px }}
details {{ background:white;padding:14px 20px;border-radius:8px;margin:18px 0 }}
pre {{ white-space:pre-wrap;font:inherit }}
@media(max-width:600px) {{ main {{ padding:14px }} section {{ padding:10px }} nav a {{ padding:4px 11px }} }}
</style>
<main>
<h1>Qwen Image 2.1 · 8组版本对照</h1>
<p>左侧为当前产品 dfca7cd6，右侧为上一产品提交 ed9552e5。每组共用提示词、文本 seed 和出图 seed。1088 × 1920，8步，CFG 1.2，thinking 开启。</p>
<p>点击对照图可打开全分辨率图片，原图链接保留生成工作流信息。</p>
<nav aria-label="对照图导航">{navigation}</nav>
<a href="overview.png" target="_blank"><img src="overview.png" alt="8组对照图概览"></a>
<details><summary>原始提示词</summary><pre>{prompt}</pre></details>
{''.join(sections)}
</main></html>'''
(review / "index.html").write_text(page)
with zipfile.ZipFile(args.root / "qwen21-product-ab-16-originals.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(review.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(review))
print(review / "index.html")
print(args.root / "qwen21-product-ab-16-originals.zip")
