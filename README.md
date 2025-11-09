# 🔬 Cell R-CNN Qt: The Complete GUI Toolkit for Cell Segmentation 🔬

Welcome to **Cell R-CNN Qt**! This is a powerful and user-friendly desktop application designed for researchers and developers to **train, detect, and evaluate** instance segmentation models for cell analysis.

Built on top of the robust **Mask R-CNN** framework, this application provides a complete graphical interface, eliminating the need for complex command-line operations for most tasks. From data preparation to model validation, everything is integrated into one convenient toolkit.

![App Screenshot](https://raw.githubusercontent.com/min20120907/Cell_RCNN_Qt/master/data/demo.png)

---

## ✨ Full Feature Set

This application is more than just a detector; it's a complete pipeline!

* **🧠 End-to-End Training:** Configure, run, and monitor the training of your own Mask R-CNN models directly within the app.
* **🚀 One-Click Detection:** Load a model and images to perform instance segmentation with ease.
* **📊 Model Evaluation:** Includes scripts to calculate the **Mean Average Precision (mAP)** to validate your model's performance.
* **🔄 Data Conversion Tools:**
    * Convert **ImageJ `.roi` files** into the **COCO `.json` format** required for training.
    * Batch processing capabilities for converting and detecting large sets of images.
* **📝 Annotation Helper:** Tools to assist in creating and visualizing masks and annotations.
* **⚙️ Profile Management:** Save and load your training configurations and file paths for reproducible experiments.

---

## 🛠️ Tech Stack

* **GUI Framework:** PyQt5
* **Deep Learning:** TensorFlow 1.x & Keras
* **Core Model:** Matterport's Mask R-CNN
* **Image Processing:** OpenCV, Scikit-image
* **Python Version:** 3.6+

---

## 🚀 Getting Started: Installation

Let's get the application up and running on your system.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/min20120907/Cell_RCNN_Qt.git](https://github.com/min20120907/Cell_RCNN_Qt.git)
    cd Cell_RCNN_Qt
    ```

2.  **Set Up a Virtual Environment (Highly Recommended!)**
    ```bash
    # Create and activate the environment
    conda create --name myenv python=3.7
    conda activate myenv
    conda install cudatoolkit=11.7 cudnn=8
    ```

3.  **Install Dependencies**
    This project requires specific versions of TensorFlow and Keras. The `requirements.txt` file handles this for you.
    ```bash
    conda activate myenv
    pip install -r requirements.txt
    ```

4.  **Download Pre-trained COCO Weights**
    For transfer learning or initial testing, you need the base Mask R-CNN weights.
    * [Download `mask_rcnn_coco.h5` from the Matterport GitHub releases.](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
    * Place the downloaded `.h5` file into the **root directory** of this project.

You're all set to go! 🎉

---

## 📖 Step-by-Step Guide

### 1. How to Convert Your Data (ImageJ ROIs -> COCO JSON)

The model needs annotations in the COCO `.json` format. This tool provides a utility to convert them from ImageJ's `.roi` files.

1.  **Launch the Application:**
    ```bash
    python Cell_Trainer.py
    ```
2.  **Open the Converter:**
    * Click the **`Convert ImageJ ROIs`** button.
3.  **Select Your Files:**
    * You will be prompted to select the directory containing your **source images** (e.g., `.png` files).
    * Next, select the directory containing the corresponding **ImageJ `.roi` files**.
    * Finally, choose a location and filename to save the output **`trainval.json`** file.
4.  **Done! ✅** The script `roi2coco_line.py` will process your files and create the JSON annotation file needed for training. For batch processing, you can use the **`Batch Convert ImageJ ROIs`** button.

### 2. How to Train Your Own Model ( •̀ ω •́ )✧

1.  **Prepare Your Dataset:**
    Your dataset folder should have the following structure:
    ```
    /your_dataset_folder
    ├── train/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── trainval.json  (The COCO annotation file you just generated!)
    ```
2.  **Launch the Application** and configure the training parameters:
    * **`Confidence Rate`**: Set the detection confidence threshold (e.g., `0.9`).
    * **`Training Steps`**: Enter the number of steps per epoch (e.g., `100`).
    * **`Training Epochs`**: Enter the total number of epochs to train for (e.g., `100`).
3.  **Load Your Data and Weights:**
    * Click **`Upload datasets`** and select your `/your_dataset_folder`.
    * Click **`Upload weights`** and choose a weights file to start from. For the first training, use the `mask_rcnn_coco.h5` file. For later trainings, you can use your own previously trained models.
4.  **Start Training!**
    * Click the big **`Train it!`** button.
    * The progress bar will update, and detailed logs will appear in the text box below. Your trained models will be saved in the `mrcnn/logs` directory. The core logic is handled by `Cell_Trainer.py`.

### 3. How to Detect Cells in Images 📸

1.  **Launch the Application.**
2.  **Load the Model:**
    * Click **`Upload weights`** and select your trained model `.h5` file (from the `mrcnn/logs/...` directory).
3.  **Load Images:**
    * Click **`Upload detection images`** and select one or more images you want to analyze.
4.  **Run Detection!**
    * Click the **`Detect it!`** button. (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧
    * The application will process the images, and the results will be saved in a `results` folder. The detection logic is managed by `detectingThread.py`. For processing an entire folder, use the **`Batch Detect`** button.

### 4. How to Validate Your Model (Calculate mAP) 📊

This project includes a script to evaluate your model's performance using the Mean Average Precision (mAP) metric. This is a command-line-based process.

1.  **Prepare Your Validation Set:**
    You need a validation dataset structured similarly to your training set, with ground-truth annotations in a `.json` file.

2.  **Run the Evaluation Script:**
    Open your terminal (make sure your virtual environment is activated) and run the `eval_model.py` script. You will need to provide paths to your model and dataset.
    ```bash
    python eval_model_gpu_cell.py --weights="/path/to/your/trained_model.h5" --dataset="/path/to/your/validation_dataset"
    ```
3.  **Analyze the Results:**
    The script will output the mAP scores for different IoU (Intersection over Union) thresholds, giving you a quantitative measure of your model's accuracy.

---

## 📜 License

Distributed under the MIT License. See the `LICENSE` file for more information.

## ⭐ Show Your Support

If you find this project useful, please give it a star on GitHub! 🌟 It helps a lot!

Happy Segmenting! (^_<)〜☆
