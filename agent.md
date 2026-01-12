要求：

请你先阅读segmentation_old/下的代码 了解其功能 随后按照下面的要求 修改 重构代码。

我现在要重构这个项目的代码 在segmentation_old/下 他原来用的是MMOpenLab的一系列工作 我想更换成一些更轻量的 主要的目标就是 能有详细且方便的配置（不是命令行） 每次实验能有详细的记录 代码简单易读 且方便修改 扩展 能适应多种任务 就是科研上常见的需求和做法。
本项目是用生成式模型辅助判别式模型（分类或segmentation）进行TTA的任务。但是他的代码是用MMOpenLab写的 太复杂了 我计划用hydra。我的模型是有判别式模型和生成式模型 然后去做TTA。
记录使用W&B

**无需实现任何的分类的内容！** 专注分割。

之前的sd3的代码可以参考。


模型层：

- 判别式模型（分割模型）采用：HuggingFace Transformers 中带的

  ```
  model = SegformerForSemanticSegmentation.from_pretrained(
      "nvidia/segformer-b5-finetuned-ade-640-640"
  )
  processor = SegformerImageProcessor.from_pretrained(
      "nvidia/segformer-b5-finetuned-ade-640-640"
  )
  ```

- 生成模型采用sd3，文字生图模型。stabilityai/stable-diffusion-3-medium-diffusers。

- 参考之前的代码内容和Loss等，但是不要照抄，其中可能有错误，你需要完全阅读，然后自己重写逻辑。

要求：

- 加载图像的时候，只做初步的处理，比如缩放等。这里不要参考之前的代码，只能处理横着的图像。把图像的**短边**缩放到512，随后进行分割，seg的输出图应为4倍下采样。这里要处理竖着的图片的情况。随后后续的不同处理交给不同的模型及其配置。
- 生成模型与判别模型的图像预处理、输入等可能不同，在各自的代码中：discriminative、generative中做各自的预处理。combined用于合并模型。
- 比如判别模型的预处理在discriminative.py及其配置文件中处理。生成模型的预处理 包括裁剪、归一化等，交给generative/sd3.py及其配置文件。
- 预处理流程：
  - 分割模型：先加载缩放后的图像 用imagenet的均值归一化 然后直接给判别模型即可

  - 生成模型：先加载缩放后的图像 滑动窗口 对应logits滑动 随后归一化到[-1,1] 经过VAE pipeline等

- 生成模型应该采用滑动窗口。输入图像和seg模型输出的logit都应该进行相应尺寸的滑动，随后把图像经过VAE下采样，logit滑动后也下采样。**滑动窗口尺寸为512，应正好符合缩放后的短边尺寸，你应该在长边的尺寸滑动而不是固定在水平方向上滑动**，修改的这部分的逻辑。
- 采用batch_size=1，避免显存问题。
- **TTA时设置模型均处于eval状态**
- 使用wandb记录必要的内容。
- **生成模型的滑动窗口要完全重写，确保正确与性能，尽量避免for循环而是使用tensor的计算方式。**
- Loss：目前的状态应该是用滑动窗口的logits 下采样到latent的尺寸 然后把预测的速度 按照logits的位置 按像素加权 随后做MSE。**梯度部分 可以不用参考之前那样 把判别式模型和生成模型的梯度分开 这没有意义。直接计算并backward即可**。确保梯度的正确流动 可以同时更新判别模型和生成模型。
- 不要每次都重复计算text embedding，先计算好后复用。可以考虑保存到本地后读取，文件名要体现prompt。
- 在本地写好之后不用尝试运行，本地没有环境。你必须保证你的代码是正确的。
- 指标主要采用mIoU作为记录指标，但是其他的也要记录，就像之前的代码中那样，确保mIoU的计算是正确的。在TTA的过程中 在进度条中显示这个mIoU的均值，用torch metrics计算并记录到wandb。
- 使用 dispatch model 将sd3 pipe 的 transformer 分割到所有运行的GPU上，保持一个均衡的状态（可以兼容不同数量的显卡，自动分配），不是数据并行，而是**我是给一个 CUDA_VISIBLE_DEVICES=0,1 让程序在若干块卡上切分运行，并给出使用方式。** 这是最重要的。注意处理好数据的device问题。**必须确保模型数据的设备一致性！同时有配置项，控制是否开启gradient checkpointing。**
- **核心的topk选取与Loss计算的方式不变。**
- **精度采用bf16 mixed 混合精度 确保不会出错。**
- log部分 在配置的config中可以设置实验名称 用于本地的与wandb的记录。
- 如果不是连续的测试，那么每个task之间重置模型。
- wandb上记录的指标 应该添加实验的序号与名称 否则会互相覆盖。
- 数据读取要shuffle。
- 我的数据目录 他是这样的：

  - image_dir = self.data_root / "ADE20K_val-c" / corruption / str(severity) / "validation"

  - annotation_dir = self.data_root / "annotations" / "validation"

  - 数据路径中间有损失的类别 以及  损失程度。默认跑15个类别+最严重的5级损失。请确保annotation的读取是正确的
- 配置中 tta 里 新增一个配置项：forward_mode 可以选择是正常的TTA 还是只跑判别模型 不跑生成模型 给一个None的loss反传 不影响模型 用于对比
- 最后写README.md 文档说明如何使用。



**重点：**

1. **正确使用dispatch model**，处理数据device问题。自动分配最大的transformer。
2. 精度问题，采用bf16混合。
3. 不要使用lightning等框架，会与dispatch model不兼容。
