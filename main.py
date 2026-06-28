import os
import csv
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


class CorruptTransform:
    def __init__(self, corruption="clean", severity=1):
        self.corruption = corruption
        self.severity = severity
        self.normalize = T.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )

    def __call__(self, img):
        x = TF.to_tensor(img)

        if self.corruption == "noise":
            std = 0.05 * self.severity
            x = x + torch.randn_like(x) * std
            x = torch.clamp(x, 0, 1)

        elif self.corruption == "brightness":
            factor = 1.0 + 0.25 * self.severity
            x = torch.clamp(x * factor, 0, 1)

        elif self.corruption == "dark":
            factor = max(0.1, 1.0 - 0.15 * self.severity)
            x = torch.clamp(x * factor, 0, 1)

        elif self.corruption == "blur":
            kernel_size = 3 + 2 * self.severity
            x = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size])

        elif self.corruption == "clean":
            pass

        elif self.corruption == "contrast":

            factor = max(
                0.2,
                1.0 - 0.15*self.severity
            )

            mean = x.mean()

            x = torch.clamp(
                (x-mean)*factor + mean,
                0,
                1
            )

        elif self.corruption == "fog":

            fog = (
                torch.randn_like(x)*0.05
                + 0.2*self.severity
            )

            x = torch.clamp(
                x + fog,
                0,
                1
            )

        elif self.corruption == "snow":

            snow = (
                torch.rand_like(x)
                < 0.02*self.severity
            ).float()

            x = torch.clamp(
                x + snow,
                0,
                1
            )

        elif self.corruption == "jpeg":

            levels = {
                1:32,
                3:16,
                5:8
            }

            q = levels[self.severity]

            x = torch.round(x*q)/q

        elif self.corruption == "pixelate":

            scale = {
                1:2,
                3:4,
                5:8
            }[self.severity]

            h,w = x.shape[1:]

            x = F.interpolate(
                x.unsqueeze(0),
                size=(h//scale,w//scale),
                mode="nearest"
            )

            x = F.interpolate(
                x,
                size=(h,w),
                mode="nearest"
            ).squeeze(0)

        else:
            raise ValueError(f"Unknown corruption: {self.corruption}")

        return self.normalize(x)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward_features(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, return_features=False):
        feat = self.forward_features(x)
        logits = self.backbone.fc(feat)

        if return_features:
            return logits, feat

        return logits


def get_train_loader(batch_size=128):
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )


def get_test_loader(corruption="clean", severity=1, batch_size=128):
    transform = CorruptTransform(corruption=corruption, severity=severity)

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )


def train_source(args):
    device = get_device()
    print("Device:", device)

    os.makedirs("./checkpoints", exist_ok=True)

    model = ResNet18CIFAR().to(device)
    train_loader = get_train_loader(args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()

        correct = 0
        total = 0

        pbar = tqdm(train_loader)

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            acc = 100.0 * correct / total

            pbar.set_description(
                f"Epoch {epoch + 1} Loss {loss.item():.4f} Acc {acc:.2f}"
            )

        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch + 1} Final Acc: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), "./checkpoints/source_resnet18.pth")
    print("Model saved to ./checkpoints/source_resnet18.pth")


def configure_model_for_tent(model):
    model.train()
    model.requires_grad_(False)

    # FC layer

    for p in model.backbone.fc.parameters():
        p.requires_grad = True
        
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    params = []
    params.extend(
    list(model.backbone.fc.parameters())
    )

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)

    return params


@torch.no_grad()
def compute_prototypes(model, loader, device, num_classes=10):
    model.eval()

    feature_sum = None
    counts = torch.zeros(num_classes).to(device)

    for x, y in tqdm(loader, desc="Computing prototypes"):
        x = x.to(device)
        y = y.to(device)

        _, feat = model(x, return_features=True)

        if feature_sum is None:
            feature_sum = torch.zeros(num_classes, feat.size(1)).to(device)

        for c in range(num_classes):
            mask = (y == c)

            if mask.sum() > 0:
                feature_sum[c] += feat[mask].sum(dim=0)
                counts[c] += mask.sum()

    prototypes = feature_sum / counts.unsqueeze(1).clamp(min=1)
    prototypes = F.normalize(prototypes, dim=1)

    return prototypes


@torch.no_grad()
def evaluate_source(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Source"):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total


def evaluate_tent(model, loader, device, lr=1e-3):
    params = configure_model_for_tent(model)
    optimizer = torch.optim.Adam(params, lr=lr)

    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="TENT"):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = entropy_loss(logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total


def evaluate_topology_tent(
    model,
    loader,
    prototypes,
    device,
    lr=1e-3,
    lambda_topo=10.0
):
    params = configure_model_for_tent(model)
    optimizer = torch.optim.Adam(params, lr=lr)

    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Confidence-Weighted Topology-TENT"):
        x = x.to(device)
        y = y.to(device)

        logits, feat = model(x, return_features=True)

        # entropy loss
        ent_loss = entropy_loss(logits)

        # confidence
        prob = F.softmax(logits, dim=1)
        confidence, pred = prob.max(dim=1)

        # normalized feature
        feat = F.normalize(feat, dim=1)

        # prototype
        target_proto = prototypes[pred]

        # ---------- topology loss ----------

        sample_topo_loss = (
            1.0
            - F.cosine_similarity(
                feat,
                target_proto,
                dim=1
            )
        )

        # ---------- simplified BTW ----------

        class_counts = torch.bincount(
            pred,
            minlength=10
        ).float()

        weights = (
            class_counts.sum()
            /
            (class_counts + 1e-6)
        )

        weights = weights / weights.mean()

        sample_weights = weights[pred]

        weighted_topo_loss = (
            sample_weights.detach()
            * confidence.detach()
            * sample_topo_loss
        ).mean()

        # ---------- intra compactness ----------

        intra_loss = 1.0

        unique_classes = pred.unique()

        for c in unique_classes:

            mask = pred == c

            if mask.sum() > 1:

                class_feat = feat[mask]

                center = class_feat.mean(
                    dim=0,
                    keepdim=True
                )

                intra_loss += (
                    (class_feat - center)
                    .pow(2)
                    .sum(dim=1)
                    .mean()
                )

        if len(unique_classes) > 0:
            intra_loss /= len(unique_classes)

        # ---------- final loss ----------

        loss = (
            ent_loss
            + lambda_topo * weighted_topo_loss
            + 0.2 * intra_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100.0 * correct / total

def evaluate_topology_tent_plus(
    model,
    loader,
    prototypes,
    device,
    lr=1e-3,
    lambda_topo=10.0
):
    return evaluate_topology_tent(
        model,
        loader,
        prototypes,
        device,
        lr,
        lambda_topo
    )

def save_results_csv(results, filename="results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corruption",
                "severity",
                "source",
                "tent",
                "topology_tent"
            ]
        )
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"Saved CSV to {filename}")


def plot_average_accuracy(results, filename="average_accuracy.png"):
    source_avg = np.mean([r["source"] for r in results])
    tent_avg = np.mean([r["tent"] for r in results])
    topo_avg = np.mean([r["topology_tent"] for r in results])

    methods = [
        "Source",
        "TENT",
        "Topology-TENT+"
    ]
    values = [source_avg, tent_avg, topo_avg]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(methods, values)

    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Average Accuracy under Distribution Shift", fontsize=14)
    plt.ylim(0, max(values) + 10)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.2f}",
            ha="center",
            fontsize=10
        )

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {filename}")


def plot_corruption_comparison(results, filename="corruption_comparison.png"):
    grouped = {}

    for r in results:
        corruption = r["corruption"]

        if corruption == "clean":
            continue

        if corruption not in grouped:
            grouped[corruption] = {
                "source": [],
                "tent": [],
                "topology_tent": []
            }

        grouped[corruption]["source"].append(r["source"])
        grouped[corruption]["tent"].append(r["tent"])
        grouped[corruption]["topology_tent"].append(r["topology_tent"])

    corruptions = list(grouped.keys())

    source_vals = [np.mean(grouped[c]["source"]) for c in corruptions]
    tent_vals = [np.mean(grouped[c]["tent"]) for c in corruptions]
    topo_vals = [np.mean(grouped[c]["topology_tent"]) for c in corruptions]

    x = np.arange(len(corruptions))
    width = 0.25

    plt.figure(figsize=(9, 5))

    plt.bar(x - width, source_vals, width, label="Source")
    plt.bar(x, tent_vals, width, label="TENT")
    plt.bar(x + width, topo_vals, width, label="Topology-TENT+")

    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Corruption Type", fontsize=12)
    plt.title("Accuracy Comparison across Corruption Types", fontsize=14)

    plt.xticks(x, corruptions)
    plt.ylim(0, max(topo_vals + tent_vals + source_vals) + 10)

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {filename}")


def run_experiments(args):
    device = get_device()
    print("Device:", device)

    ckpt = "./checkpoints/source_resnet18.pth"

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            "Please train source model first: python main.py --mode train"
        )

    base_model = ResNet18CIFAR().to(device)
    base_model.load_state_dict(torch.load(ckpt, map_location=device))

    train_loader = get_train_loader(args.batch_size)

    prototypes = compute_prototypes(
        base_model,
        train_loader,
        device
    )

    corruptions = [
        "clean",
        "noise",
        "brightness",
        "dark",
        "blur",
        "contrast",
        "fog",
        "snow",
        "jpeg",
        "pixelate"
    ]

    severities = [1, 3, 5]

    results = []

    print("\n================ RESULTS ================")
    print(
        "Corruption | Severity | "
        "Source | TENT | Topology-TENT+"
    )

    for corruption in corruptions:
        for severity in severities:
            if corruption == "clean" and severity != 1:
                continue

            loader = get_test_loader(
                corruption=corruption,
                severity=severity,
                batch_size=args.batch_size
            )

            source_model = ResNet18CIFAR().to(device)
            source_model.load_state_dict(torch.load(ckpt, map_location=device))

            tent_model = ResNet18CIFAR().to(device)
            tent_model.load_state_dict(torch.load(ckpt, map_location=device))

            topo_model = ResNet18CIFAR().to(device)
            topo_model.load_state_dict(torch.load(ckpt, map_location=device))

            source_acc = evaluate_source(
                source_model,
                loader,
                device
            )

            tent_acc = evaluate_tent(
                tent_model,
                loader,
                device,
                lr=args.adapt_lr
            )

            topo_acc = evaluate_topology_tent(
                topo_model,
                loader,
                prototypes,
                device,
                lr=args.adapt_lr,
                lambda_topo=args.lambda_topo
            )

            print(
                f"{corruption:10s} | "
                f"{severity:8d} | "
                f"{source_acc:6.2f} | "
                f"{tent_acc:6.2f} | "
                f"{topo_acc:13.2f}"
            )

            results.append({
                "corruption": corruption,
                "severity": severity,
                "source": round(source_acc, 4),
                "tent": round(tent_acc, 4),
                "topology_tent": round(topo_acc, 4)
            })

    save_results_csv(results, "results.csv")
    plot_average_accuracy(results, "average_accuracy.png")
    plot_corruption_comparison(results, "corruption_comparison.png")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--adapt_lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--lambda_topo",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "train":
        train_source(args)
        args.mode = "eval"
    
    if args.mode == "eval":
        run_experiments(args)


if __name__ == "__main__":
    main()