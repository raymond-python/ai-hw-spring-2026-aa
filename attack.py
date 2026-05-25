import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    adv_images = images + epsilon * images.grad.sign()
    return torch.clamp(adv_images, 0, 1).detach()

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    ori_images = images.clone().detach().to(device)
    adv_images = ori_images.clone().detach()

    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        adv_images = adv_images + alpha * adv_images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach()

    return adv_images

def mifgsm_attack(model, images, labels, epsilon, alpha, iters, decay=1.0):
    ori_images = images.clone().detach().to(device)
    adv_images = ori_images.clone().detach()
    momentum = torch.zeros_like(images).to(device)

    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = adv_images.grad
        grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = decay * momentum + grad

        adv_images = adv_images + alpha * momentum.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(ori_images + eta, 0, 1).detach()

    return adv_images

def evaluate_attack(model, test_loader, attack_name, attack_func):
    total_clean_correct = 0
    total_adv_success = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        clean_outputs = model(images)
        clean_preds = clean_outputs.argmax(dim=1)
        clean_correct_mask = clean_preds == labels

        adv_images = attack_func(model, images, labels)
        adv_outputs = model(adv_images)
        adv_preds = adv_outputs.argmax(dim=1)

        successful_attack = clean_correct_mask & (adv_preds != labels)

        total_clean_correct += clean_correct_mask.sum().item()
        total_adv_success += successful_attack.sum().item()
        total += labels.size(0)

    recognition_rate = total_clean_correct / total
    asr = total_adv_success / total_clean_correct

    print(f"{attack_name}: Recognition Rate = {recognition_rate*100:.2f}%, ASR = {asr*100:.2f}%")

    return {
        "attack": attack_name,
        "recognition_rate_before_attack": recognition_rate,
        "attack_success_rate": asr
    }

transform = transforms.ToTensor()
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

model = MNISTCNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
model.eval()

results = []

eps = 0.3
alpha = 0.01
iters = 40

results.append(evaluate_attack(
    model,
    test_loader,
    "FGSM",
    lambda m, x, y: fgsm_attack(m, x, y, eps)
))

results.append(evaluate_attack(
    model,
    test_loader,
    "I-FGSM / PGD",
    lambda m, x, y: pgd_attack(m, x, y, eps, alpha, iters)
))

results.append(evaluate_attack(
    model,
    test_loader,
    "Momentum I-FGSM",
    lambda m, x, y: mifgsm_attack(m, x, y, eps, alpha, iters)
))

df = pd.DataFrame(results)
df.to_csv("results/attack_results.csv", index=False)
df.to_json("results/attack_results.json", orient="records", indent=4)

print("Results saved to results/")
