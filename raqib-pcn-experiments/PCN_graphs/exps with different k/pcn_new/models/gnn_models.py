import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import ChebConv, EdgeConv

class PCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k):
        super(PCN, self).__init__()
        self.conv1 = ChebConv(in_feats, hidden_feats, k)
        self.conv2 = ChebConv(hidden_feats, hidden_feats, k)
        self.edge_conv = EdgeConv(hidden_feats, hidden_feats)
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, g):
        h = F.relu(self.conv1(g, g.ndata["feat"]))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.edge_conv(g, h))
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")
        return self.fc(hg)

class PCN_Lite(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k):
        super(PCN_Lite, self).__init__()
        self.conv1 = ChebConv(in_feats, hidden_feats, k)
        self.conv2 = ChebConv(hidden_feats, hidden_feats, k)
        self.fc = nn.Linear(hidden_feats, out_feats)

    def forward(self, g):
        h = F.relu(self.conv1(g, g.ndata["feat"]))
        h = F.relu(self.conv2(g, h))
        g.ndata["h"] = h
        hg = dgl.mean_nodes(g, "h")
        return self.fc(hg)