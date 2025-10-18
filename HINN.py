import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__()
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
        # Learnable scaling weights, initialized to 1
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

        # 1. SNP to CpG layer (Genetic influence on methylation)
        self.snp_to_cpg = MaskedLinear(snp_dim, cpg_dim, mask_snp_cpg, bias=True)
        self.g_cpg = FeatureScaler(cpg_dim)
        self.B2 = nn.Parameter(torch.zeros(cpg_dim))

        # 2. CpG to Gene Expression layer (Epigenetic regulation of expression)
        self.cpg_to_gene = MaskedLinear(cpg_dim, gene_dim, mask_cpg_gene, bias=True)
        self.g_gene = FeatureScaler(gene_dim)
        self.B3 = nn.Parameter(torch.zeros(gene_dim))

        # 3. Gene Expression to GO Term layer (Functional grouping)
        self.gene_to_go = MaskedLinear(gene_dim, go_dim, mask_gene_go, bias=True)

        # --- Fully Connected Shortcut Layers ---
        self.snp_fc20 = nn.Linear(snp_dim, fc_nodes)
        self.cpg_fc20 = nn.Linear(cpg_dim + fc_nodes, fc_nodes)
        self.gene_fc20 = nn.Linear(gene_dim + fc_nodes, fc_nodes)

        # --- Dense Layers for Prediction ---
        dense_input_dim = go_dim + fc_nodes
        self.dense_block1 = self._build_dense_block(dense_input_dim, hidden_dense_units, dropout_prob)
        self.dense_block2 = self._build_dense_block(hidden_dense_units, hidden_dense_units, dropout_prob)
        self.dense_block3 = self._build_dense_block(hidden_dense_units, hidden_dense_units, dropout_prob)
        self.dense_block4 = self._build_dense_block(hidden_dense_units, hidden_dense_units, dropout_prob)
        
        self.post_dense20 = nn.Linear(hidden_dense_units, 20)

        # --- Demographic Data Integration ---
        demog_input_dim = 20 + demog_dim
        self.demog_block = self._build_dense_block(demog_input_dim, hidden_dense_units, dropout_prob)
        
        # --- Output Layer ---
        self.output_layer = nn.Linear(hidden_dense_units, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_dense_block(self, in_features, out_features, dropout_prob):
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x_snp, x_cpg, x_gene, demog, eps=1e-8):
        # Layer 1
        Y1 = F.relu(self.snp_to_cpg(x_snp))

        # Layer 2
        gX2 = self.g_cpg(x_cpg)
        Y2 = F.relu((gX2 * Y1) + self.B2)

        # Layer 3
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

def load_and_prepare_data(target_column='ADAS11'):
    try:
        demo_df = pd.read_csv('demo_label_data.csv')
        gene_df = pd.read_csv('gene_data.csv')
        methyl_df = pd.read_csv('methyl_data.csv')
        snp_df = pd.read_csv('snp_data.csv')
        
        mask_snp_cpg_df = pd.read_csv('snp_methyl_matrix.csv', index_col=0)
        mask_cpg_gene_df = pd.read_csv('methyl_gene_matrix.csv', index_col=0)
        mask_gene_go_df = pd.read_csv('gene_pathway_matrix.csv', index_col=0)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure all CSV files are in the same directory.")
        return None

    data = pd.merge(demo_df, snp_df, on='IID', how='inner')
    data = pd.merge(data, methyl_df, on='IID', how='inner')
    data = pd.merge(data, gene_df, on='IID', how='inner')

    demog_cols = ['AGE', 'GENDER', 'EDU', 'RACE', 'PTAU181', 'APOE']
    
    valid_targets = ['MMSE', 'MOCA', 'ADAS11', 'RAVLT.immediate']
    if target_column not in valid_targets:
        raise ValueError(f"Invalid target_column. Choose from: {valid_targets}")

    y = data[target_column].values
    X_demog = data[demog_cols].values
    
    snp_features = [col for col in snp_df.columns if col != 'IID']
    cpg_features = [col for col in methyl_df.columns if col != 'IID']
    gene_features = [col for col in gene_df.columns if col != 'IID']
    go_features = list(mask_gene_go_df.columns)

    X_snp = data[snp_features].values
    X_cpg = data[cpg_features].values
    X_gene = data[gene_features].values



    mask_snp_cpg = mask_snp_cpg_df.loc[snp_features, cpg_features].values
    mask_cpg_gene = mask_cpg_gene_df.loc[cpg_features, gene_features].values

    mask_gene_go = mask_gene_go_df.loc[gene_features, go_features].values

    scaler = StandardScaler()
    X_demog_scaled = scaler.fit_transform(X_demog)

    X_snp_tensor = torch.FloatTensor(X_snp)
    X_cpg_tensor = torch.FloatTensor(X_cpg)
    X_gene_tensor = torch.FloatTensor(X_gene)
    X_demog_tensor = torch.FloatTensor(X_demog_scaled)
    y_tensor = torch.FloatTensor(y)
    
    mask_snp_cpg_tensor = torch.from_numpy(mask_snp_cpg).float()
    mask_cpg_gene_tensor = torch.from_numpy(mask_cpg_gene).float()
    mask_gene_go_tensor = torch.from_numpy(mask_gene_go).float()
    
    return X_snp_tensor, X_cpg_tensor, X_gene_tensor, X_demog_tensor, y_tensor, mask_snp_cpg_tensor, mask_cpg_gene_tensor, mask_gene_go_tensor

if __name__ == '__main__':
    target_variable = 'ADAS11' 
    
    data_tensors = load_and_prepare_data(target_column=target_variable)
    
    if data_tensors is None:
        print("Data loading failed. Exiting.")
        exit()

    X_snp, X_cpg, X_gene, X_demog, y, mask_snp_cpg, mask_cpg_gene, mask_gene_go = data_tensors
    
    print(f"Number of samples: {len(y)}")
    
    indices = list(range(len(y)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    X_snp_train, X_snp_test = X_snp[train_indices], X_snp[test_indices]
    X_cpg_train, X_cpg_test = X_cpg[train_indices], X_cpg[test_indices]
    X_gene_train, X_gene_test = X_gene[train_indices], X_gene[test_indices]
    X_demog_train, X_demog_test = X_demog[train_indices], X_demog[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    train_dataset = TensorDataset(X_snp_train, X_cpg_train, X_gene_train, X_demog_train, y_train)
    test_dataset = TensorDataset(X_snp_test, X_cpg_test, X_gene_test, X_demog_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    snp_dim = X_snp.shape[1]
    cpg_dim = X_cpg.shape[1]
    gene_dim = X_gene.shape[1]
    go_dim = mask_gene_go.shape[1]
    demog_dim = X_demog.shape[1]

    model = HINN(
        snp_dim=snp_dim,
        cpg_dim=cpg_dim,
        gene_dim=gene_dim,
        go_dim=go_dim,
        mask_snp_cpg=mask_snp_cpg,
        mask_cpg_gene=mask_cpg_gene,
        mask_gene_go=mask_gene_go,
        demog_dim=demog_dim
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (snp, cpg, gene, demog, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(snp, cpg, gene, demog)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        

    model.eval()
    total_mae = 0
    total_mse = 0
    with torch.no_grad():
        for snp, cpg, gene, demog, labels in test_loader:
            outputs = model(snp, cpg, gene, demog)
            mse = criterion(outputs, labels)
            mae = F.l1_loss(outputs, labels) # Mean Absolute Error
            
            total_mse += mse.item() * len(labels)
            total_mae += mae.item() * len(labels)

    avg_mse = total_mse / len(test_dataset)
    avg_mae = total_mae / len(test_dataset)
    
    print(f'Mean Squared Error (MSE) on the test set: {avg_mse:.4f}')
    print(f'Mean Absolute Error (MAE) on the test set: {avg_mae:.4f}')
