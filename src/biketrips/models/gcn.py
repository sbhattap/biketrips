import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import BallTree
from torch import nn
from torch_geometric.nn import GCNConv

from biketrips.models import Model


class Net(nn.Module):
    """Two-layer GCN with a linear head; the head outputs a log rate."""

    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        self.conv1, self.conv2 = GCNConv(n_in, hidden), GCNConv(hidden, hidden)
        self.fc, self.dropout = nn.Linear(hidden, 1), dropout

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), self.dropout, self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), self.dropout, self.training)
        return self.fc(h).squeeze(-1).clamp(max=15)  # exp(15) is far above any cell's count


class GCN(Model):
    """Transductive: every cell is a node; only training labels are used. Test cells stay in the graph
    with their urban features visible and their labels masked."""

    def _graph(self, cells, train_idx, k):
        x = np.log1p(cells[self.features].to_numpy(float))
        med = np.nanmedian(x[train_idx], axis=0)  # imputation and scaling use training nodes only
        x = np.where(np.isnan(x), med, x)
        mu, sd = x[train_idx].mean(0), x[train_idx].std(0) + 1e-9
        _, nbr = BallTree(np.radians(cells[["lat", "lng"]].to_numpy()), metric="haversine").query(
            np.radians(cells[["lat", "lng"]].to_numpy()), k=k + 1)
        src, dst = nbr[:, 1:].ravel(), np.repeat(np.arange(len(cells)), k)  # neighbours -> node
        edges = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=self.device)
        # Rare categories standardise to 40+ sd on a few nodes, which overflows exp() in the first epochs.
        x = np.clip((x - mu) / sd, -5, 5)
        return torch.tensor(x, dtype=torch.float, device=self.device), edges

    def fit(self, X, y):
        p, cells = self.params, self.params["cells"]
        torch.manual_seed(self.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.node = {h: i for i, h in enumerate(cells["h3"])}
        idx = np.array([self.node[h] for h in X["h3"]])
        # Early-stopping nodes are whole res-6 groups held out of the training nodes.
        groups = X["group"].unique()
        rng = np.random.default_rng(self.seed)
        val_groups = rng.choice(groups, max(1, round(p["val_frac"] * len(groups))), replace=False)
        is_val = X["group"].isin(val_groups).to_numpy()
        self.x, self.edges = self._graph(cells, idx[~is_val], p["k"])
        target = torch.zeros(len(cells), device=self.device)
        target[idx] = torch.tensor(np.asarray(y, float), dtype=torch.float, device=self.device)
        fit_idx, val_idx = (torch.tensor(i, device=self.device) for i in (idx[~is_val], idx[is_val]))
        self.net = Net(self.x.shape[1], p["hidden"], p["dropout"]).to(self.device)
        with torch.no_grad():  # start from the mean rate so early epochs are not spent finding the scale
            self.net.fc.bias.fill_(float(np.log(target[fit_idx].mean().item() + 1)))
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"])
        loss_fn = nn.PoissonNLLLoss(log_input=True)
        best, best_state, wait = np.inf, None, 0
        for _ in range(p["max_epochs"]):
            self.net.train()
            opt.zero_grad()
            loss_fn(self.net(self.x, self.edges)[fit_idx], target[fit_idx]).backward()
            opt.step()
            self.net.eval()
            with torch.no_grad():
                val = loss_fn(self.net(self.x, self.edges)[val_idx], target[val_idx]).item()
            if val < best:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            elif (wait := wait + 1) >= p["patience"]:
                break
        self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            rate = torch.exp(self.net(self.x, self.edges)).cpu().numpy()
        return rate[[self.node[h] for h in X["h3"]]]
