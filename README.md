## Cài đặt

Yêu cầu Python 3.9 trở lên.

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows  (Linux/macOS: source .venv/bin/activate)

# 1) PyTorch có CUDA (để train trên GPU NVIDIA). Trên Windows, `pip install torch` thường
#    chỉ ra bản CPU -> lấy đúng lệnh ở https://pytorch.org/get-started/locally/ , ví dụ:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) Các thư viện còn lại
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # GPU: phải in True
```

## Dữ liệu

Tải FER-2013 vào https://www.kaggle.com/datasets/deadskull7/fer2013

## Huấn luyện M1, M2

Mở `notebooks/train.ipynb` và chạy toàn bộ notebook. Notebook làm các việc sau:
- Thống kê phân bố lớp, xử lý mất cân bằng.
- Train M1 và M2, lưu weights tốt nhất theo val macro-F1 vào `weights/`.
- Vẽ đồ thị loss/accuracy, confusion matrix, ảnh dự đoán sai (vào `results/`).
- Export ONNX, đo tốc độ và tạo bảng so sánh M3 (`results/m3_comparison.csv`).


## Chạy ứng dụng

```bash
# Camera điện thoại: chỉ cần IP:cổng hiện trên app (IP Webcam: 8080 | DroidCam: 4747)
python src/app.py --source 192.168.1.31:8080 --model cnn
# URL đầy đủ cũng được
python src/app.py --source http://192.168.1.31:8080/video --model cnn
# Ảnh tĩnh (kết quả lưu vào outputs/)
python src/app.py --source test.jpg --model transfer
# File video / webcam laptop
python src/app.py --source clip.mp4 --model cnn
python src/app.py --source 0 --model cnn
# Không truyền --source -> dùng camera.source trong config.yaml
python src/app.py --model cnn
```

Hoặc mở `notebooks/demo.ipynb`, sửa dòng `IP = "..."` ở cell đầu rồi chạy từng cell.

Nếu gặp lỗi `The function is not implemented ... namedWindowImpl`, môi trường đang cài OpenCV bản headless (không mở được cửa sổ). Sửa bằng 2 lệnh sau:
```bash
pip uninstall -y opencv-python-headless opencv-contrib-python-headless opencv-contrib-python opencv-python
pip install opencv-python
```

| Tham số | Ý nghĩa |
|---|---|
| `--source` | URL IP camera, đường dẫn ảnh/video, hoặc số webcam |
| `--model cnn \| transfer` | M1 hoặc M2 |
| `--backend onnx \| torch` | Mặc định ONNX (nhanh hơn trên CPU) |
| `--conf 0.4` | Ngưỡng confidence (dưới ngưỡng: nhãn có "?", khung xám) |
| `--save out.mp4` | Lưu kết quả ra file |
| `--show-crops` | Hiện ảnh 48×48 thật sự đưa vào mô hình (phím `c` để bật/tắt) |

Phím tắt: `q` hoặc `ESC` để thoát, `s` để chụp màn hình vào `outputs/`.

**Xử lý lỗi:**
- Mất kết nối camera: app hiện "CAMERA DISCONNECTED", tự kết nối lại khi điện thoại có mạng trở lại, không crash, không treo.
- File không tồn tại, ảnh hỏng, ảnh không có khuôn mặt, thiếu weights: app in thông báo rõ ràng và thoát an toàn.

### Kết nối điện thoại

1. Cài IP Webcam (Android) hoặc DroidCam, chọn **Start server**, ghi lại địa chỉ, ví dụ `http://192.168.1.15:8080`.
2. Mở địa chỉ đó trên trình duyệt laptop để kiểm tra thấy hình.
3. Trong app điện thoại, chỉnh độ phân giải 640×480 để giảm độ trễ và tăng FPS.
4. Không kết nối được thì kiểm tra:
   - Hai máy có cùng Wi-Fi không. Nếu dùng hotspot điện thoại thì laptop bắt hotspot đó.
   - URL đã có `/video` ở cuối chưa.
   - Tường lửa Windows có chặn Python không.

## Kết quả

Đánh giá trên **FER-2013 PrivateTest** (3.589 ảnh). Tập train có 28.697 ảnh (đã loại 12 ảnh trơn), tập val (PublicTest) có 3.589 ảnh.

| Mô hình | Accuracy | Macro-F1 | Tham số | ONNX | CPU PyTorch | CPU ONNX | FPS model-only | FPS end-to-end* | Train (GPU) |
|---|---|---|---|---|---|---|---|---|---|
| M1 CNN tự thiết kế | **70,86%** | **0,695** | 4,82M | 18,4 MB | 8,20 ms | 4,97 ms | 201 | **29,2** | 12,5 phút |
| M2 ResNet18 (TL) | **71,30%** | **0,704** | 11,18M | 42,6 MB | 14,95 ms | 5,17 ms | 193 | **≈30–40** | 7,2 phút |

Ngưỡng của đề là M1 ≥ 60% và M2 ≥ 65%. Cả hai mô hình đều đạt.

\* FPS end-to-end gồm cả đọc camera, phát hiện mặt, mô hình và vẽ. Đo với IP Webcam ở độ phân giải 640×480, laptop cắm sạc. M1 là trung bình của 2.188 khung hình; M2 là giá trị hiển thị trên khung hình trong lúc demo. Cả hai bị giới hạn bởi tốc độ phát của camera điện thoại (khoảng 30 khung/giây), không phải bởi mô hình.

F1 theo từng lớp:

| | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
|---|---|---|---|---|---|---|---|
| M1 | 0,642 | 0,732 | 0,511 | 0,908 | 0,574 | 0,803 | 0,693 |
| M2 | 0,659 | 0,752 | 0,546 | 0,903 | 0,556 | 0,832 | 0,684 |

Nhận xét chính:
- M2 chỉ hơn M1 0,44 điểm accuracy, tương đương khoảng 16 ảnh trên 3.589. Mức chênh này nhỏ hơn sai số chuẩn của phép đo (khoảng ±0,76%), nên **hai mô hình ngang nhau về độ chính xác**.
- Chạy bằng ONNX trên CPU, M2 chỉ chậm hơn M1 khoảng 0,2 ms. Lý do là input 112×112 được giữ nhỏ.
- Fear và Sad là hai lớp khó nhất (F1 khoảng 0,51–0,57). Các cặp nhầm nhiều nhất giống nhau ở cả hai mô hình: Sad → Neutral (21%), Fear → Sad (17%), Disgust → Angry (15–16%), Fear → Angry (12–15%).
- Disgust tuy chỉ chiếm 1,5% dữ liệu train nhưng vẫn đạt F1 khoảng 0,73–0,75, cho thấy class weight có tác dụng.

Hình chi tiết (confusion matrix, đồ thị huấn luyện, ảnh dự đoán sai) nằm trong `results/` và `report.pdf`.

## Cấu trúc

```
├── README.md
├── HYPERPARAMS.md        # giải thích siêu tham số + câu hỏi vấn đáp
├── requirements.txt
├── config.yaml           # URL camera, đường dẫn model, ngưỡng confidence, detector
├── notebooks/
│   ├── train.ipynb       # huấn luyện + đánh giá M1, M2 (M3, M4, M5), đã chạy, còn output
│   └── demo.ipynb        # chạy app realtime từ notebook (chỉ cần sửa IP)
├── src/
│   ├── model.py          # M1 CNN, M2 transfer, InputAdapter, lưu/nạp, export ONNX
│   ├── detector.py       # YuNet (OpenCV DNN) / Haar
│   ├── predict.py        # crop + tiền xử lý + dự đoán 1 ảnh khuôn mặt
│   ├── app.py            # ứng dụng realtime (--source, --model)
│   ├── data.py           # đọc FER-2013 (CSV / thư mục), chia tập
│   ├── train.py          # vòng huấn luyện, augmentation trên GPU
│   ├── analysis.py       # biểu đồ, confusion matrix, ảnh sai
│   └── benchmark.py      # đo tốc độ CPU
├── weights/              # cnn.pt/.onnx, transfer.pt/.onnx, face_detection_yunet_2023mar.onnx
├── results/              # hình và số liệu do train.ipynb sinh ra (dùng trong báo cáo)
├── data/                 # đặt fer2013.csv vào đây (không kèm theo bài nộp)
├── test.jpg              # ảnh mẫu cho lệnh --source test.jpg
└── report.pdf
```

Face detector YuNet: [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT License).
