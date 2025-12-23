# @title 🚀 YOLO11 ULTIMATE TRAINING PIPELINE (Upgrade v2.0)
# @markdown ---
# @markdown ### 🎯 Cấu hình Chế độ Training
import os
import sys
import yaml
import psutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- CẤU HÌNH GIAO DIỆN COLAB ---
train_mode = "Higher Accuracy" # @param ["Fast", "Higher Accuracy", "Tune"]
model_size = "yolo11n.pt" # @param ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
dataset_yaml_path = "/content/dataset_xe/data.yaml" # @param {type:"string"}
project_name = "yolo11_project" # @param {type:"string"}

# @markdown ---
# @markdown ### ⚙️ Tùy chọn cho "Higher Accuracy"
high_acc_strategy = "1. Chay N lan (Tham so Vang)" # @param ["1. Chay N lan (Tham so Vang)", "2. Chay theo file Tune (Can chay Tune truoc)"]
# @markdown *Số lần chạy (để tìm ra model có mAP cao nhất - Chỉ áp dụng cho Option 1):*
n_runs = 1 # @param {type:"integer"}

# @markdown ---
# @markdown ### 🛠️ Tham số & Dữ liệu
dataset_quality = "Binh thuong" # @param ["Sach (Clean)", "Binh thuong", "Ban/Nhieu (Noisy)"]
img_size = 640 # @param {type:"integer"}
# @markdown *Lưu ý: Fast Mode sẽ bỏ qua Epochs này và dùng mặc định 50.*
target_epochs = 300 # @param {type:"slider", min:50, max:600, step:10}

# ==============================================================================
# PHẦN 1: CÁC CLASS HỖ TRỢ THÔNG MINH (CORE LOGIC)
# ==============================================================================

class HardwareManager:
    """Tự động kiểm tra phần cứng để tối ưu hóa Cache (Mục 3.1 Tài liệu)."""
    @staticmethod
    def get_cache_strategy():
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        available_ram_gb = mem.available / (1024 ** 3)

        print(f"🖥️ SYSTEM CHECK: RAM Available={available_ram_gb:.1f}GB / Total={total_ram_gb:.1f}GB")

        # Nếu RAM trống > 12GB (Colab Pro), dùng cache RAM để max tốc độ
        if available_ram_gb > 12:
            print("🚀 Kích hoạt: CACHE RAM (Tốc độ tối đa, giảm I/O đĩa).")
            return 'ram'
        else:
            print("💾 Kích hoạt: CACHE DISK (Tiết kiệm RAM, tránh crash).")
            return 'disk'

class SmartCallback:
    """
    Callback can thiệp quá trình train thời gian thực (Mục 6 tài liệu).
    Tự động giảm LR nếu mAP không tăng (Thông minh hơn Cosine mặc định).
    """
    def __init__(self, patience=15, decay=0.5):
        self.patience = patience
        self.decay = decay
        self.best_fitness = 0.0
        self.wait = 0

    def on_fit_epoch_end(self, trainer):
        # Lấy metrics quan trọng nhất
        metrics = trainer.metrics
        # Tìm key đúng (do các phiên bản YOLO có thể đổi tên key)
        keys = [k for k in metrics.keys() if 'map50-95' in k.lower()]
        current_map = metrics.get(keys[0], 0) if keys else 0

        if current_map > self.best_fitness + 0.0001:
            self.best_fitness = current_map
            self.wait = 0
        else:
            self.wait += 1
            # Nếu kiên nhẫn hết hạn -> Giảm LR nóng
            if self.wait >= self.patience:
                if hasattr(trainer, 'optimizer'):
                    old_lr = trainer.optimizer.param_groups[0]['lr']
                    new_lr = old_lr * self.decay
                    for g in trainer.optimizer.param_groups:
                        g['lr'] = new_lr
                    print(f"\n⚡ [AI Tuner] Phát hiện bão hòa. Giảm LR: {old_lr:.6f} -> {new_lr:.6f}")
                self.wait = 0

class Reporter:
    """Phân tích kết quả sau train và đưa ra nhận xét."""
    @staticmethod
    def analyze(save_dir):
        csv_path = os.path.join(save_dir, 'results.csv')
        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Lấy dữ liệu
        best_idx = df['metrics/mAP50(B)'].idxmax()
        best_map50 = df['metrics/mAP50(B)'].iloc[best_idx]
        best_map95 = df['metrics/mAP50-95(B)'].iloc[best_idx]
        final_box_loss = df['val/box_loss'].iloc[-1]
        train_box_loss = df['train/box_loss'].iloc[-1]
        final_dfl_loss = df['val/dfl_loss'].iloc[-1]

        print("\n" + "="*40)
        print("📊 BÁO CÁO PHÂN TÍCH HIỆU SUẤT (AI REPORT)")
        print("="*40)
        print(f"🏆 Best mAP@50:    {best_map50:.4f} (Epoch {best_idx+1})")
        print(f"🥇 Best mAP@50-95: {best_map95:.4f}")
        print(f"📉 Final Val Losses: Box={final_box_loss:.3f} | DFL={final_dfl_loss:.3f}")
        print("-" * 40)
        print("💡 NHẬN XÉT & KHUYẾN NGHỊ:")

        # Logic phân tích
        tips = []
        if train_box_loss < final_box_loss * 0.7:
            tips.append("⚠️ CÓ DẤU HIỆU OVERFITTING: Train Loss thấp hơn nhiều so với Val Loss.")
            tips.append("   -> Giải pháp: Tăng 'weight_decay', chọn Dataset Quality='Ban/Nhieu' để tăng Augmentation.")

        if final_dfl_loss > 1.8:
             tips.append("⚠️ DFL LOSS CAO: Mô hình gặp khó khăn xác định biên vật thể.")
             tips.append("   -> Giải pháp (YOLO11): Tăng tham số 'dfl' (vd: 2.0) hoặc tăng 'imgsz' lên 1280.")

        if best_map50 < 0.5:
            tips.append("⚠️ ĐỘ CHÍNH XÁC THẤP: Model chưa học được đặc trưng.")
            tips.append("   -> Giải pháp: Kiểm tra lại dataset, label, hoặc tăng số lượng Epochs.")

        if not tips:
            print("✅ Mô hình cân bằng tốt, hội tụ ổn định.")
        else:
            for tip in tips:
                print(tip)

        # Vẽ biểu đồ
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(df['train/box_loss'], label='Train Box Loss')
        ax[0].plot(df['val/box_loss'], label='Val Box Loss')
        ax[0].set_title("Loss Analysis")
        ax[0].legend()
        ax[1].plot(df['metrics/mAP50(B)'], label='mAP@50')
        ax[1].plot(df['metrics/mAP50-95(B)'], label='mAP@50-95')
        ax[1].set_title("Accuracy Analysis")
        ax[1].legend()
        plt.show()
        print("="*40 + "\n")

# ==============================================================================
# PHẦN 2: LOGIC CẤU HÌNH (THAM SỐ VÀNG)
# ==============================================================================

def get_golden_augmentations(quality):
    """Augmentation dựa trên chất lượng data (Mục 4.2 Tài liệu)"""
    base = {'mosaic': 1.0, 'fliplr': 0.5}
    if quality == "Sach (Clean)":
        # Data sạch -> Ít biến đổi để giữ đặc trưng
        base.update({'mixup': 0.0, 'degrees': 0.0, 'scale': 0.2})
    elif quality == "Ban/Nhieu (Noisy)":
        # Data nhiễu -> Biến đổi mạnh để model học tốt hơn
        base.update({'mixup': 0.2, 'degrees': 15.0, 'scale': 0.8, 'copy_paste': 0.3})
    else: # Binh thuong
        base.update({'mixup': 0.1, 'scale': 0.5})
    return base

def get_golden_hyperparams():
    """Bộ tham số vàng cho YOLO11 (Suy luận từ tài liệu)"""
    return {
        'optimizer': 'auto',     # Để YOLO tự chọn
        'lr0': 0.01,
        'lrf': 0.01,
        'cos_lr': True,          # Cosine Scheduler giúp hội tụ mượt
        'warmup_epochs': 3.0,
        'box': 7.5,              # Gain mặc định
        'cls': 0.5,
        'dfl': 1.5,              # Quan trọng cho YOLO11 (Anchor-free)
        'close_mosaic': 20,      # Tắt Mosaic 20 epoch cuối để học ảnh thật (Mục 4.3)
    }

# ==============================================================================
# PHẦN 3: CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

print(f"🚀 KHỞI ĐỘNG HỆ THỐNG: {train_mode.upper()} | Model: {model_size}")
model = YOLO(model_size)
cache_strat = HardwareManager.get_cache_strategy()
aug_config = get_golden_augmentations(dataset_quality)

# Đăng ký Callback thông minh
smart_cb = SmartCallback(patience=20)
model.add_callback("on_fit_epoch_end", smart_cb.on_fit_epoch_end)

if train_mode == "Fast":
    # === FAST MODE ===
    print("⚡ FAST MODE: Tối ưu tốc độ tối đa (Cache RAM + AutoBatch + AMP).")
    results = model.train(
        data=dataset_yaml_path,
        epochs=50,              # Giới hạn 50 epoch để nhanh
        imgsz=640,
        batch=-1,               # AutoBatch: Tự tính batch lớn nhất
        device=0,
        workers=8,
        cache=cache_strat,      # Cache thông minh
        amp=True,               # Mixed Precision (Nhanh gấp đôi trên T4/V100)
        patience=30,
        optimizer='SGD',        # SGD hội tụ nhanh hơn đoạn đầu
        project=project_name,
        name='fast_run',
        exist_ok=True,
        plots=True
    )
    Reporter.analyze(results.save_dir)

elif train_mode == "Tune":
    # === TUNE MODE ===
    print("🎶 TUNE MODE: Tìm kiếm bộ tham số vàng (Genetic Algorithm).")
    print("⏳ Quá trình này rất tốn thời gian, vui lòng kiên nhẫn...")
    # Tập trung search space vào các tham số nhạy cảm
    model.tune(
        data=dataset_yaml_path,
        epochs=30,
        iterations=20,          # Chạy 20 thử nghiệm
        optimizer='AdamW',      # AdamW tốt cho search space rộng
        plots=True,
        save=False,
        val=True,
        cache=cache_strat
    )
    print("✅ Tune hoàn tất! File tham số tại: 'runs/detect/tune/best_hyperparameters.yaml'")

elif train_mode == "Higher Accuracy":
    # === HIGHER ACCURACY MODE ===
    print("🔥 HIGHER ACCURACY MODE: Tối ưu độ chính xác cực đại.")

    final_args = {}
    loops = 1

    # 1. Xác định nguồn tham số
    if high_acc_strategy.startswith("2"): # Load từ Tune
        tune_path = 'runs/detect/tune/best_hyperparameters.yaml'
        if os.path.exists(tune_path):
            print(f"📂 Đang load tham số Tune từ: {tune_path}")
            with open(tune_path) as f:
                final_args = yaml.safe_load(f)
        else:
            print("❌ Không tìm thấy file Tune. Chuyển sang chạy Tham số Vàng mặc định.")
            high_acc_strategy = "1" # Fallback

    if high_acc_strategy.startswith("1"): # Tham số Vàng
        print("💎 Sử dụng bộ Tham số Vàng (Expert Params - Manual).")
        final_args = get_golden_hyperparams()
        final_args.update(aug_config) # Thêm augmentation vào config
        loops = n_runs # Chạy N lần theo yêu cầu

    # 2. Vòng lặp Training (N lần hoặc 1 lần)
    best_map = 0
    best_dir = ""

    for i in range(1, loops + 1):
        run_name = f'high_acc_run_{i}' if loops > 1 else 'high_acc_run'
        print(f"\n🎬 --- RUN {i}/{loops} ---")

        # Reset model để clean state
        temp_model = YOLO(model_size)
        temp_model.add_callback("on_fit_epoch_end", smart_cb.on_fit_epoch_end)

        try:
            res = temp_model.train(
                data=dataset_yaml_path,
                epochs=target_epochs,
                imgsz=img_size,
                batch=16,           # Batch ổn định cho Accuracy (thay vì Auto)
                cache=cache_strat,
                device=0,
                patience=50,        # Kiên nhẫn hơn
                save=True,
                project=project_name,
                name=run_name,
                exist_ok=True,
                **final_args        # Truyền bộ tham số vàng vào
            )

            # Đánh giá kết quả run hiện tại
            current_map = res.box.map50
            print(f"📊 KẾT QUẢ RUN {i}: mAP50 = {current_map:.4f}")

            if current_map > best_map:
                best_map = current_map
                best_dir = res.save_dir
                print("⭐ KỶ LỤC MỚI! Đây là model tốt nhất hiện tại.")

            Reporter.analyze(res.save_dir)

        except Exception as e:
            print(f"❌ Có lỗi xảy ra ở Run {i}: {e}")

    if loops > 1:
        print(f"\n🏆 TỔNG KẾT: Model mạnh nhất nằm tại '{best_dir}' với mAP50={best_map:.4f}")

print("\n✅ CHUỖI NHIỆM VỤ HOÀN TẤT.")
