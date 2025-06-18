# 🚀 Fine-Tuning a Multimodal Reward Model for Web Interactions

This repository contains the complete code and documentation for building, fine-tuning, and testing a multimodal reward model from scratch. The model is designed to assess the quality of textual descriptions for given webpage screenshots, a foundational task for developing aligned AI web agents.

The entire project is self-contained within the provided Jupyter/Colab notebook, which you can run to replicate the results.

### Project Overview & Goal

In modern AI, simply having a powerful pre-trained model is not enough. To be truly useful and safe, these models must be **aligned** with human preferences and intent. A key technique to achieve this is to train a **Reward Model (RM)** that acts as a learned "preference score" for an AI's outputs.

This project tackles that challenge head-on in a multimodal context. The goal was to take a powerful vision-language model and fine-tune it to understand a specific preference: *Does this sentence accurately describe the content of this webpage screenshot?*

The resulting reward model can then be used to rank different potential descriptions or as a crucial component in a larger Reinforcement Learning from Human Feedback (RLHF) pipeline.

---

### The ML Pipeline: A Look at the Components

This project was structured to touch every critical stage of a modern machine learning pipeline.

#### **1. Data Collection & Curation**
The foundation of any model is its data. Instead of using a generic dataset, we created our own by programmatically capturing screenshots of live, diverse websites using browser automation. This raw visual data was then curated into a high-quality **preference dataset**. For each screenshot, a "good" (`chosen`) description and a "bad" (`rejected`) description were manually created, forming the basis for our preference tuning.

#### **2. Model Architecture & Technical Choices**

The core of this project lies in the specific choices made for the model and training process.

* **Why LLaVA 1.5 (7B)?**
    LLaVA (Large Language and Vision Assistant) is a powerful open-source model that excels at understanding both images and text simultaneously. It combines a pre-trained **CLIP vision encoder** with a **LLaMA-based language model**. This architecture was chosen because it provides a strong, pre-trained foundation for visual reasoning, which is essential for understanding the content of a webpage screenshot.

* **Why a Reward Model?**
    Instead of fine-tuning the model to generate text (a standard SFT task), we re-purposed it to output a single scalar value—a reward score. We achieved this by "monkey-patching" the model: adding a new, trainable linear layer (the `reward_head`) to its architecture. This new head takes the model's deep understanding of the image and text and condenses it into a single number representing quality.

* **Why the Log Sigmoid Loss Function?**
    This is the standard loss function for preference tuning, made famous by OpenAI's InstructGPT. Its goal is not to make the reward score match a specific number, but to enforce the preference `Score(chosen) > Score(rejected)`. It works by converting the difference in scores into a probability using a sigmoid function and then using a log loss to maximize that probability. This is a robust and effective way to teach the model a relative preference.

* **Why 4-bit Quantization and Other Optimizations?**
    Fine-tuning a 7-billion-parameter model is incredibly memory-intensive. To make this feasible on a single cloud GPU (like a Google Colab A100), several state-of-the-art memory-saving techniques were essential:
    * **4-bit Quantization (`bitsandbytes`):** The model's weights were loaded as 4-bit integers instead of 16-bit floats, reducing the memory footprint by nearly 75%.
    * **Gradient Checkpointing:** This technique trades computation for memory by not storing all intermediate activations during the forward pass, drastically reducing the peak memory usage.
    * **`device_map="auto"` (`accelerate`):** Intelligently splits the model across the GPU and CPU, ensuring it fits in memory.

#### **3. Training & Evaluation**
A manual PyTorch training loop was implemented to provide full control over the process, which was necessary to debug and stabilize the complex interaction between the quantized model and the hardware. At the end of each epoch, a complete checkpoint—containing the base model, the trained reward head, and the processor configuration—was saved to ensure reproducibility. The final model is then evaluated in the testing portion of the notebook, where it demonstrates its learned ability to assign a higher score to accurate descriptions over inaccurate ones.

### How to Use This Project

The beauty of this project is its simplicity. All you need to do is:

1.  **Open the `.ipynb` file** in Google Colab.
2.  **Connect to a GPU runtime** .
3.  **Mount your Google Drive** when prompted.
4.  **Run the cells in order**, from top to bottom.

The notebook will handle all installations, data creation, training, and testing, guiding you through the entire process.
