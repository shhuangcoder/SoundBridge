# **Sound Bridge：association Egocentric and Exocentric videos via Audio Cues**<br>
Almost there🤭



## 📢 News
**[2025.3.10]** The code and dataset of related tasks has been released.

**[2025.3.5]** The repository is public.

**[2025.2.28]** The repository is created.

<a name="installation"></a>
## ⚙️ Installation
1. Clone the repository from GitHub.

```shell
git clone https://github.com/shhuangcoder/SoundBridge.git
cd SoundBridge
```

2. Create conda environment.

```shell
conda create -n SoundBridge python=3.8
conda activate SoundBridge
```


3. Download the packages
```shell
pip install -r requirements.txt
```

## 🗂️ Dataset
For the dataset, we provide the corresponding download link. Please follow the download instructions provided in the link to download the dataset.<br>
EgoExoLearn: https://github.com/OpenGVLab/EgoExoLearn<br>
CharadesEgo: https://prior.allenai.org/projects/charades-ego<br>
You need to first extract the audio file (.wav) from the video and then use the BEATs model for audio feature extraction. Please call the BEATs model according to its requirements. [BEATS](https://github.com/microsoft/unilm.git)

### The prepared dataset should be in the following structure.
```
.
├── SoundBridge
│   ├── model
│   └── data
│   └── results
│   └── config
│   └── annotations
│   └── utils
│   └── function
├── datasets
│   └── Charades_Ego
│   │   └── videos
│   │  └── audios
│   └── EgoExoLearn
│   │   └── videos
│   │   └── audios
├── ckpt
│   └── Cha.pth
│   └── EgoExo.pth
├──README.md
└── ···
```


## 🪐 Checkpoint
* You can download from [huggingface](https://huggingface.co/Sihong/SoundBridge)


