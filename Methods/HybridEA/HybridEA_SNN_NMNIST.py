import os
import gc
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import snntorch as snn
import wandb
import tonic
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil


# ------------------------
# 0. Basic env & warnings
# ------------------------
os.environ["WANDB_MODE"] = "offline"   # set to "online" to sync live
warnings.filterwarnings("ignore", category=RuntimeWarning, module="tonic")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", DEVICE)

# --------------------------
# 1. Dataset Loader (NMNIST)
# --------------------------
NUM_NEURONS = 34 * 34  # 34x34 pixels
TIME_BINS   = 300      # NMNIST time bins shall be longer
NUM_CLASSES = 10       # digit 0–9

class NMNIST_Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.data = tonic.datasets.NMNIST(save_to="./tonic_data", train=train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ev, label = self.data[idx]
        t, x, y = ev["t"], ev["x"], ev["y"]
        t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        dense = torch.zeros(TIME_BINS, NUM_NEURONS)
        if len(t):
            idx_t = (t / t.max() * (TIME_BINS - 1)).astype(int)
            idx_xy = (y * 34 + x).astype(int)
            dense[idx_t, idx_xy] = 1.0
        return dense, label

TRAIN_DS = NMNIST_Dataset(train=True)
TEST_DS  = NMNIST_Dataset(train=False)

# --------------------------
# 2. Sparse SNN definition
# --------------------------
class SNN(nn.Module):
    def __init__(self, nin=NUM_NEURONS, nhid=100, nout=NUM_CLASSES, beta=0.95, sparsity=0.8):
        super().__init__()
        self.fc1 = nn.Linear(nin, nhid, bias=False)
        self.fc2 = nn.Linear(nhid, nout, bias=False)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta, reset_mechanism="none")
        # sparsify weights
        with torch.no_grad():
            for w in (self.fc1.weight, self.fc2.weight):
                w.mul_((torch.rand_like(w) > sparsity).float())

    def forward(self, x):  # x: [B,T,N]
        T = x.size(1)
        x = x.permute(1, 0, 2)  # [T,B,N]
        mem1 = torch.zeros(x.size(1), self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(x.size(1), self.fc2.out_features, device=x.device)
        outs = []
        for t in range(T):
            spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            _,   mem2 = self.lif2(self.fc2(spk1), mem2)
            outs.append(mem2)
        return torch.stack(outs)  # [T,B,C]

    # flat param helpers
    def flat(self):
        return torch.cat([p.view(-1) for p in self.parameters()])

    def load_flat(self, vec: torch.Tensor):
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel(); p.copy_(vec[idx:idx+n].view_as(p)); idx += n

# --------------------------
# 3. Hybrid ES (PSO + Pool)
# --------------------------
CE = nn.CrossEntropyLoss()

def hybrid_update(model, mean, vel, p_best, g_best, std, k, xb, yb, lr=0.01, acc_thr=0.90):
    pop = mean + std * torch.randn(k, mean.numel(), device=DEVICE)  # Sample population
    losses, accs = [], []

    # Evaluate each sampled particle
    for vec in pop:
        model.load_flat(vec)
        out = model(xb).mean(0)
        losses.append(CE(out, yb).item())
        accs.append((out.argmax(1) == yb).float().mean().item())
    
    losses = torch.tensor(losses, device=DEVICE)
    accs = torch.tensor(accs, device=DEVICE)

    # Update global best
    best_idx = losses.argmin()
    model.load_flat(mean)
    curr_loss = CE(model(xb).mean(0), yb).item()
    if losses[best_idx] < curr_loss:
        g_best = pop[best_idx].clone()

    # PSO-inspired velocity update
    r1, r2 = torch.rand(2, device=DEVICE)
    vel = 0.5 * vel + 1.5 * r1 * (p_best - mean) + 1.5 * r2 * (g_best - mean)
    mean = mean + lr * vel

    # Evaluate new mean after PSO update
    model.load_flat(mean)
    out_now = model(xb).mean(0)
    loss_now = CE(out_now, yb)
    acc_now = (out_now.argmax(1) == yb).float().mean().item()

    # Maintain personal best (only update if better)
    model.load_flat(p_best)
    pbest_loss = CE(model(xb).mean(0), yb).item()
    if loss_now.item() < pbest_loss:
        p_best = mean.clone()

    # Adaptive Pooling with Offspring (if needed)
    if accs[best_idx] < acc_thr:
        topk_idx = losses.argsort()[:k // 4]
        parent = pop[topk_idx]
        offspring = parent + std * torch.randn_like(parent)  # Generate offspring

        offspring_losses = []
        for vec in offspring:
            model.load_flat(vec)
            out = model(xb).mean(0)
            offspring_losses.append(CE(out, yb).item())
        offspring_losses = torch.tensor(offspring_losses, device=DEVICE)

        best_offspring_idx = offspring_losses.argsort()[:max(1, parent.size(0) // 2)]
        mean = offspring[best_offspring_idx].mean(0).detach()

        print("Adaptive Pooling with Offspring", end=" ")

        # After pooling, re-evaluate mean and update p_best again if needed
        model.load_flat(mean)
        out_now = model(xb).mean(0)
        loss_now = CE(out_now, yb)
        acc_now = (out_now.argmax(1) == yb).float().mean().item()

        model.load_flat(p_best)
        pbest_loss = CE(model(xb).mean(0), yb).item()
        if loss_now.item() < pbest_loss:
            p_best = mean.clone()

    # Logging
    print(f"Batch Acc: {acc_now*100:.1f}%", flush=True)
    wandb.log({"train_acc": acc_now, "train_loss": loss_now.item()}, commit=False)

    return mean, vel, p_best, g_best, loss_now.item()

# --------------------------
# 4. Training loop
# --------------------------

def train():
    cfg = dict(nhid=100, epochs=100, batch_size=256, pop=800, std0=0.12, lr=0.01)
    wandb.init(entity='DarwinNeuron', project='EA-NMNIST', name="hybrid_ES_offline_low_decay", config=cfg)

    tr_loader = DataLoader(TRAIN_DS, batch_size=cfg['batch_size'], shuffle=True)
    val_loader= DataLoader(TEST_DS , batch_size=256, shuffle=True)

    model = SNN(nhid=cfg['nhid']).to(DEVICE)
    mean = model.flat().clone(); vel = torch.zeros_like(mean); p_best=g_best=mean.clone()
    val_acc_prev = 0
    global_step, best_val_loss = 0, float("inf")
    best_path = "best_nmnist_model.pth"

    for ep in range(cfg['epochs']):
        print(f"[RAM] {psutil.virtual_memory().used / 1024**3:.2f} GB used")
        print(f"Epoch {ep}"); 
        # adaptive exploration logic
        if val_acc_prev > 0.5:
            std = cfg['std0'] * (0.995 ** (ep - acc_50_epoch))
            k = max(100, int(cfg['pop'] * (0.995 ** (ep - acc_50_epoch))))
        else:
            std = cfg['std0']
            k = cfg['pop']
        acc_thr = 0.9+0.08*ep/cfg['epochs']
        for xb, yb in tr_loader:
            mean, vel, p_best, g_best, val_acc_prev = hybrid_update(model, mean, vel, p_best, g_best, std, k,
                                                       xb.to(DEVICE).float(), yb.to(DEVICE), lr=cfg['lr'], acc_thr=acc_thr)
            global_step += 1 # for every batch +1
            wandb.log({"global_step": global_step})  # set batch count as x-axis
        # ---- validation & firing ----
        model.load_flat(mean)
        with torch.no_grad():
            val_loss_sum, total, correct = 0.0,0,0;fire_sum=torch.zeros(cfg['nhid'], device=DEVICE)
            for xv,yv in val_loader:
                xv=xv.to(DEVICE).float(); yv=yv.to(DEVICE)
                out = model(xv).mean(0); val_loss_sum += CE(out, yv).item() * yv.size(0)
                correct += (out.argmax(1) == yv).sum().item(); total += yv.size(0)
                mem=torch.zeros(xv.size(0), cfg['nhid'], device=DEVICE)
                for t in range(TIME_BINS):
                    spk, mem = model.lif1(model.fc1(xv.permute(1,0,2)[t]), mem)
                    fire_sum+=spk.sum(0)
            val_loss = val_loss_sum / total; val_acc=correct/total; fire_avg=(fire_sum/total).cpu().numpy()
        if val_acc_prev <= 0.5 and val_acc > 0.5:
            acc_50_epoch = ep
            print(f">>> Reached 50% acc at epoch {ep}, exploration will now decay.")
        val_acc_prev = val_acc

        # heatmap
        fig=plt.figure(figsize=(10,2)); sns.heatmap(fire_avg[np.newaxis,:], cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
        plt.title(f"Hidden firing Epoch {ep}"); wandb.log({"firing_heatmap": wandb.Image(fig)})
        plt.close(fig)
        wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "epoch": ep, "std": std, "samples": k, "hidden_firing_mean": fire_avg.mean()})
        print(f"Validation Accuracy: {val_acc:.4f} | std: {std:.4f} | samples: {k} | acc_thresh: {acc_thr:.4f} | hidden_firing_mean: {fire_avg.mean()}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"[Checkpoint] epoch {ep}  val_loss ↓ to {val_loss:.4f}  (saved)")
        
        torch.cuda.empty_cache(); gc.collect()

    wandb.finish()

if __name__ == "__main__":
    train()

"""
# To load saved model:
model = SNN(nhid=cfg['nhid']).to(DEVICE)
state = torch.load("best_model.pth", map_location=DEVICE)
model.load_state_dict(state)
model.eval()
"""
