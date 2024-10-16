# clothesMEA

의류 키포인트 추출 및 의류 파트별 길이 예측 구현 테스트



## 테스트 준비사항

### [모델 체크포인트](https://huggingface.co/Seoksee/MY_MODEL_FILE/tree/main)

- .gitattributes 를 제외한 모든 파일을 src/model에 넣기


### [DeepFashion2 데이터 세트 (선택사항)](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok)

-  `HRNet-for-Fashion-Landmark-Estimation.PyTorch`의 데이터 세트

### CUDA Toolkit 11.8 설치 (선택사항)

- Ubuntu 22.04

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    apt-get install cuda-toolkit-11.8

    #설치 완료 후 환경변수 추가
    ===
    [편집기] ~/.bashrc
        #가장 아래에 추가
        export PATH=/usr/local/cuda/bin:$PATH 
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ===
    source ~/.bashrc
    ```

## 테스트 방법

1. **Conda 설치**

2. **가상환경 생성**

    ```bash
    conda env create -p .conda

    # cuda 사용
    conda env create -f environment_cuda.yml -p .conda
    ```

3. **프로젝트 Python 파일 테스트**

## 모델 주석

### 의류 키포인트 추정
-   [HRNet-for-Fashion-Landmark-Estimation.PyTorch](https://github.com/Lseoksee/HRNet-for-Fashion-Landmark-Estimation.PyTorch)의 예측 코드 추출 버전

### 카드 감지

- [Yolov8](https://docs.ultralytics.com/models/yolo11/) 모델에 자체 제작 데이터 세트 훈련