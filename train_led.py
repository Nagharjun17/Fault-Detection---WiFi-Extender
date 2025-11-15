import os, json, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import multiprocessing as mp

def main():
    ROOT = Path.home() / "Downloads" / "board_dataset_work"
    DATA = ROOT / "dataset"
    OUT  = ROOT / "models"
    OUT.mkdir(parents=True, exist_ok=True)

    n_on_tr  = len(list((DATA/"train"/"LED_ON").glob("*.jpg")))
    n_off_tr = len(list((DATA/"train"/"LED_OFF").glob("*.jpg")))
    n_on_va  = len(list((DATA/"val"/"LED_ON").glob("*.jpg")))
    n_off_va = len(list((DATA/"val"/"LED_OFF").glob("*.jpg")))
    print(f"train ON={n_on_tr} OFF={n_off_tr} | val ON={n_on_va} OFF={n_off_va}")
    if (n_on_tr + n_off_tr + n_on_va + n_off_va) == 0:
        raise RuntimeError("No images found. Re-run make_dataset.sh or check paths.")

    IMG_SIZE = 224
    BATCH   = 32
    EPOCHS  = 15
    LR      = 1e-3
    NUM_CLASSES = 2
    CLASS_NAMES = ["LED_OFF", "LED_ON"]

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    trn_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(DATA / "train", transform=trn_tfms)
    val_ds   = datasets.ImageFolder(DATA / "val",   transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=0, pin_memory=False, persistent_workers=False)

    model = timm.create_model("mobilenetv2_100", pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_acc, best_path = 0.0, OUT / "mobilenetv2_led_best.pth"

    def evaluate():
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss_sum += loss.item() * yb.size(0)
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return loss_sum / total, correct / total

    for epoch in range(1, EPOCHS+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = evaluate()
        print(f"Epoch {epoch:02d}: val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "class_names": ["LED_OFF","LED_ON"],
                        "img_size": IMG_SIZE,
                        "normalize_mean": [0.485,0.456,0.406],
                        "normalize_std":  [0.229,0.224,0.225]}, best_path)

    torch.save({"model": model.state_dict(),
                "class_names": ["LED_OFF","LED_ON"],
                "img_size": IMG_SIZE,
                "normalize_mean": [0.485,0.456,0.406],
                "normalize_std":  [0.229,0.224,0.225]},
               OUT / "mobilenetv2_led_last.pth")
    print(f"Best val_acc={best_acc:.3f}; saved {best_path}")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    torch.set_num_threads(1)
    main()
