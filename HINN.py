import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__()
        assert mask.shape == (in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.05)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('mask', mask.to(torch.float32))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = self.weight * self.mask
        out = x @ w
        if self.bias is not None:
            out = out + self.bias
        return out

class FeatureScaler(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.ws = nn.Parameter(torch.ones(features))

    def forward(self, x):
        return x * self.ws

class HINN(nn.Module):
    def __init__(self,
                 snp_dim,
                 cpg_dim,
                 gene_dim,
                 go_dim,
                 mask_snp_cpg,
                 mask_cpg_gene,
                 mask_gene_go,
                 demog_dim=6,
                 fc_nodes=20,
                 hidden_dense_units=128,
                 dropout_prob=0.7):
        super().__init__()

        self.snp_to_cpg = MaskedLinear(snp_dim, cpg_dim, mask_snp_cpg, bias=True)
        self.g_cpg = FeatureScaler(cpg_dim)
        self.B2 = nn.Parameter(torch.zeros(cpg_dim))

        self.cpg_to_gene = MaskedLinear(cpg_dim, gene_dim, mask_cpg_gene, bias=True)
        self.g_gene = FeatureScaler(gene_dim)
        self.B3 = nn.Parameter(torch.zeros(gene_dim))

        self.gene_to_go = MaskedLinear(gene_dim, go_dim, mask_gene_go, bias=True)

        self.snp_fc20 = nn.Linear(snp_dim, fc_nodes)
        self.cpg_fc20 = nn.Linear(cpg_dim + fc_nodes, fc_nodes)
        self.gene_fc20 = nn.Linear(gene_dim + fc_nodes, fc_nodes)

        dense_input_dim = go_dim + fc_nodes
        self.dense_block1 = self._build_dense_block(dense_input_dim, hidden_dense_units, dropout_prob)
        self.dense_block2 = self._build_dense_block(hidden_dense_units, hidden_dense_units, dropout_prob)
        self.dense_block3 = self._build_dense_block(hidden_dense_units, hidden_dense_units, dropout_prob)
        self.dense_block4 = self._build_dense_block(hidden_dense_units, hidden_dense_units, dropout_prob)
        
        self.post_dense20 = nn.Linear(hidden_dense_units, 20)

        demog_input_dim = 20 + demog_dim
        self.demog_block = self._build_dense_block(demog_input_dim, hidden_dense_units, dropout_prob)
        self.output_layer = nn.Linear(hidden_dense_units, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_dense_block(self, in_features, out_features, dropout_prob):
        return nn.Sequential(
            nn.BatchNorm1d(in_features, momentum=0.9, eps=0.005),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

    def forward(self, x_snp, x_cpg, x_gene, demog, eps=1e-8):
        Y1 = F.relu(self.snp_to_cpg(x_snp))

        gX2 = self.g_cpg(x_cpg)
        Y2 = F.relu((gX2 * Y1) + self.B2)

        gX3 = self.g_gene(x_gene)

        fY2 = self.cpg_to_gene(Y2)
        denom = fY2 + eps
        Y3 = F.relu((gX3 / denom) + self.B3)

        Y4 = F.relu(self.gene_to_go(Y3))

        snp_fc_out = F.relu(self.snp_fc20(x_snp))
        
        cpg_fc_in = torch.cat([Y2, snp_fc_out], dim=1)
        cpg_fc_out = F.relu(self.cpg_fc20(cpg_fc_in))
        
        gene_fc_in = torch.cat([Y3, cpg_fc_out], dim=1)
        gene_fc_out = F.relu(self.gene_fc20(gene_fc_in))

        dense_input = torch.cat([Y4, gene_fc_out], dim=1)
        
        h = self.dense_block1(dense_input)
        h = self.dense_block2(h)
        h = self.dense_block3(h)
        h = self.dense_block4(h)

        h20 = F.relu(self.post_dense20(h))

        demog_in = torch.cat([h20, demog], dim=1)
        demog_h = self.demog_block(demog_in)
        
        out = self.output_layer(demog_h)
        return out.squeeze(-1)
