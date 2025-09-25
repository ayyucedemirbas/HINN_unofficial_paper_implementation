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
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.features = features
        self.ws = nn.Parameter(torch.ones(features))
        self.eps = eps

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
                 demog_dim=6,  # APOE4, pTau, age, gender, education, race
                 hidden_dense_units=128,
                 dropout_prob=0.7):
        super().__init__()

        self.mask_snp_cpg = mask_snp_cpg
        self.mask_cpg_gene = mask_cpg_gene
        self.mask_gene_go = mask_gene_go

        self.snp_to_cpg = MaskedLinear(snp_dim, cpg_dim, mask_snp_cpg, bias=True)

        self.g_cpg = FeatureScaler(cpg_dim)
        self.cpg_dense20 = nn.Linear(cpg_dim, 20)

        self.cpg_to_gene = MaskedLinear(cpg_dim, gene_dim, mask_cpg_gene, bias=True)

        self.g_gene = FeatureScaler(gene_dim)
        self.gene_dense20 = nn.Linear(gene_dim + 20, 20)

        self.gene_to_go = MaskedLinear(gene_dim, go_dim, mask_gene_go, bias=True)

        self.go_dense20 = nn.Linear(go_dim + 20, 20)

        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dense_units, momentum=0.9, eps=0.005)
                                        for _ in range(4)])
        self.dense_layers = nn.ModuleList([nn.Linear(hidden_dense_units if i>0 else (go_dim + 20), hidden_dense_units)
                                           for i in range(4)])
        self.dropout = nn.Dropout(dropout_prob)
        self.post_dense20 = nn.Linear(hidden_dense_units, 20)

        self.demog_bn = nn.BatchNorm1d(20 + demog_dim, momentum=0.9, eps=0.005)
        self.demog_dense = nn.Linear(20 + demog_dim, hidden_dense_units)
        self.final_dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(hidden_dense_units, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_snp, x_cpg, x_gene, demog, eps=1e-6):
        Y1 = F.relu(self.snp_to_cpg(x_snp))

        #DNA methylation (CpG) layer
        gX2 = self.g_cpg(x_cpg)
        Y2_pre = gX2 * Y1
        Y2 = F.relu(Y2_pre)
        cpg_fc20 = F.relu(self.cpg_dense20(gX2))

        fY2 = F.relu(self.cpg_to_gene(Y2))

        #Gene layer
        gX3 = self.g_gene(x_gene)
        denom = fY2 + eps
        Y3_pre = gX3 / denom
        Y3 = F.relu(Y3_pre)
        gene_fc_in = torch.cat([Y3, cpg_fc20], dim=1)
        gene_fc20 = F.relu(self.gene_dense20(gene_fc_in))

        #GO layer
        Y4 = F.relu(self.gene_to_go(Y3))
        go_fc_in = torch.cat([Y4, gene_fc20], dim=1)
        go_fc20 = F.relu(self.go_dense20(go_fc_in))

        dense_input = torch.cat([Y4, go_fc20], dim=1)
        h = dense_input
        for bn, dense in zip(self.bn_layers, self.dense_layers):
            h = dense(h)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

        h20 = F.relu(self.post_dense20(h))

        demog_in = torch.cat([h20, demog], dim=1)
        demog_in = self.demog_bn(demog_in)
        demog_h = F.relu(self.demog_dense(demog_in))
        demog_h = self.final_dropout(demog_h)

        out = self.output_layer(demog_h)
        return out.squeeze(-1)
