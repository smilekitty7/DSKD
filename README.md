
## Dynamically Semantic-Guided Knowledge Distillation for Incremental Object Detection

Incremental object detection (IOD) task requires a model to learn continually from newly added data. However, directly fine-tuning a well-trained detection model on a new task will sharply decrease the performance on the old task, which is known as catastrophic forgetting.
Knowledge distillation, including feature distillation and response distillation, has been proved to be an effective way to alleviate catastrophic forgetting. However, previous works on feature distillation heavily rely on low-level feature selection, while under-explore the importance of high-level semantic information. In this paper, we propose a method that dynamically distills semantic information with consideration of its discriminativeness and consistency. Between-class discriminativeness is preserved by distilling semantic distance among various categories, while within-class consistency is preserved by dynamically distilling feature maps under the guidance of semantic information. Extensive experiments are conducted on MS COCO benchmarks. The performance of our method exceeds previous SOTA methods under all experimental scenarios. Remarkably, our method reduces the mAP gap toward full-training to 1.0, which is much better than that of the previous SOTA method with a gap of 3.3.

## Transformer & Incremental Object Detection
Our method is the first to implement knowledge distillation in Transformer structure for full-dataset incremental object detection, with the help of structure property of attention mechanism, more meaningful semantic information and the useful query-based bipartite matching process. Our work demonstrates an impressive prospective of Transformer structure in incremental (continual) learning.

## Our Contributions
 - **To the best of our knowledge, we are the first to explicitly explore the within-class and between-class knowledge distillation for incremental object detection task.**
 
 - **We propose a novel feature distillation method based on the dynamic interaction between high-level semantics and low-level feature to keep the within-class consistency for incremental object detection task.**

 - **We propose a new between-class difference distillation method based on distance matrix of high-level semantic feature to keep the between-class discriminativeness.**

## Overall Structure
![image](https://img-blog.csdnimg.cn/9a3433b021224a7a83c5c83157bf67f7.png)
The overall framework of the proposed method is shown in the figure. 
 - **Between-Class Distance Distillation**

    We use this module to preserve the between-class discrimnativeness.

- **Dynamically Semantic-Guided Feature Distillation**

    We use this module to preserve the within-class consistency. 

## Results

We tested our methods under different experimental settings, like 40 classes+ 40 classes, 50 classes+ 30 classes, 60 classes+ 20 classes, and 70 classes+ 10 classes. Results are shown below. 

![image](https://img-blog.csdnimg.cn/86bc76f740a646f6b1c6651f79d586f9.png)
###### (ERD is proposed in CVPR 2022: [Overcoming catastrophic forgetting in incremental object detectionn via elastic response distillation](https://arxiv.org/abs/2204.02136))

All these results show that our method exceeds the current SOTA method, ERD, demonstrating that our method has better capability of reducing catastrophic forgetting. Further details can be found in our paper. 
 
 
