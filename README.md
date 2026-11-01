# NexStitch

5대의 카메라로 촬영한 ProRes (.mov) 영상을 스티칭하는 툴.

## 폴더 구조

```
NexStitch/
├── input/          소스 영상 5개를 여기에 넣으세요
├── output/         스티칭 결과물이 저장됩니다
├── calibration/    캘리브레이션 데이터 저장
├── app.py          Streamlit 메인 앱
├── requirements.txt
└── README.md
```

## 요구 사항

### Python

Python 3.10 이상 필요.

### ffmpeg (별도 설치 필수)

OpenCV만으로는 ProRes 코덱(.mov)을 디코딩할 수 없는 경우가 있습니다.
ffmpeg는 반드시 시스템에 별도 설치하고 PATH에 등록해야 합니다.

**Windows 설치 방법 (winget)**

```powershell
winget install --id Gyan.FFmpeg -e
```

또는 [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) 에서 Windows 빌드를 받아
압축을 풀고 `bin/` 폴더를 시스템 PATH에 추가하세요.

설치 확인:

```bash
ffmpeg -version
```

### Python 라이브러리 설치

```bash
cd C:\NexStitch
pip install -r requirements.txt
```

> `opencv-contrib-python`에는 스티칭에 필요한 추가 알고리즘(SURF 등)이 포함되어 있습니다.

## 실행

```bash
cd C:\NexStitch
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 이 자동으로 열립니다.

## 사용 방법

1. `input/` 폴더에 카메라 5대의 .mov 파일을 넣습니다.
2. 앱을 실행하면 파일이 자동 감지되고 첫 프레임 미리보기가 표시됩니다.
3. 5개 파일이 모두 감지되고 ffmpeg가 설치된 경우 **스티칭 시작** 버튼이 활성화됩니다.
4. 결과물은 `output/` 폴더에 저장됩니다.
