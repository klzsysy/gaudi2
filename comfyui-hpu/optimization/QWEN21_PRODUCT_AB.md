# Qwen Image 2.1产品版本人工对照

2026-11-15，用户确认比较当前产品`dfca7cd672240da975983214c11d3057ec5b74b6`与其上一提交`ed9552e581acfa37c6be2ac6f1a86a629f21d542`，每版独立生成8张，供人工检查。

## 条件与结果

沿用用户原始植物人像提示词及最新2MP工作流，1088×1920、官方Turbo 8步sigmas、Euler、CFG1.2、thinking开启及现行BF16组件。文本seed从447606998181262至447606998181269，图像seed从614878163710363至614878163710370，两侧逐组配对。

16份history均确认TextGenerate和SamplerCustomAdvanced重新执行。8组扩写文本逐字相同，8对RGBA像素逐元素相同，全部像素MAE与最大绝对差为0。用户于2026-11-15完成人工检查，确认质量正常，接受产品`dfca7cd6`并要求保持该提交。

实验在独立checkout上用原生`main.py`入口、独立8190进程串行执行。切换的4个文件来自两个git提交，包括2个产品文件及2个测试文件，逐份SHA256核对。每个版本的请求ID、两类seed、源文件哈希和像素哈希保存在[证据](../evidence/qwen21-product-ab-2026-11-15.json)。

初版缓存检查错误地包含KSamplerSelect，第二张完成后触发断言。检查已改为按class_type识别TextGenerate和SamplerCustomAdvanced，重新核验前两张有效结果后从第3张续跑。当前版本第1/3张含进程cold费用，各请求时延保留在证据中。

## 人工检查文件

- 本地目录为`comfyui-hpu/results/qwen21-product-ab-2026-11-15/review`，含16张原PNG、各自扩写文本、8张原尺寸并排对照图、概览、HTML浏览页及manifest。
- 每组左侧为`dfca7cd6`，右侧为`ed9552e5`。并排图添加版本标题，每侧保留原尺寸像素，并用ImageMagick AE=0校验。概览为缩略图。
- `qwen21-product-ab-16-originals.zip`包含上述检查材料，ZIP完整性和HTML全部文件链接已验证。
- 服务器目录为`/home/sonny/gaudi2/comfyui-hpu/quality-qwen21-product-ab-2026-11-15`。原生成文件另位于独立checkout的`output/gaudi/qwen21_product_ab/current`和`previous`。

生成入口为`probes/qwen21_product_ab.py`，整理入口为`probes/qwen21_product_review.py`。脚本使用同参数自然扩写及采样，保存原PNG后制作确定性并排图。版本切换与异常退出均恢复当前产品。

最终普通8189服务运行`dfca7cd6`，4份文件SHA256与当前提交一致，队列为空。原始`/home/sonny/ComfyUI`遵循只读约束。
