import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import plotly.graph_objects as go

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
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
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

def plot_sankey(model, snp_names, cpg_names, gene_names, go_names, top_k=4):
    w_snp_cpg = torch.abs(model.snp_to_cpg.weight.data).numpy()
    w_cpg_gene = torch.abs(model.cpg_to_gene.weight.data).numpy()
    w_gene_go = torch.abs(model.gene_to_go.weight.data).numpy()

    mask_snp_cpg = model.snp_to_cpg.mask.numpy()
    mask_cpg_gene = model.cpg_to_gene.mask.numpy()
    mask_gene_go = model.gene_to_go.mask.numpy()

    def get_links(source_names, target_names, weights, mask, layer_top_k):
        links = []
        rows, cols = np.where(mask == 1)
        for r, c in zip(rows, cols):
            links.append({
                'source': source_names[r],
                'target': target_names[c],
                'value': weights[r, c]
            })

        return sorted(links, key=lambda x: x['value'], reverse=True)[:layer_top_k]

    top_links = []
    top_links.extend(get_links(snp_names, cpg_names, w_snp_cpg, mask_snp_cpg, top_k))
    top_links.extend(get_links(cpg_names, gene_names, w_cpg_gene, mask_cpg_gene, top_k))
    top_links.extend(get_links(gene_names, go_names, w_gene_go, mask_gene_go, top_k))
    
    if not top_links:
        print("No connections found to plot. Are the model weights all zero?")
        return

    node_labels = set()
    for link in top_links:
        node_labels.add(link['source'])
        node_labels.add(link['target'])

    ordered_labels = [n for n in snp_names if n in node_labels] + \
                     [n for n in cpg_names if n in node_labels] + \
                     [n for n in gene_names if n in node_labels] + \
                     [n for n in go_names if n in node_labels]
    
    node_map = {name: i for i, name in enumerate(ordered_labels)}


    link_data = {'source': [], 'target': [], 'value': []}
    for link in top_links:
        if link['source'] in node_map and link['target'] in node_map:
            link_data['source'].append(node_map[link['source']])
            link_data['target'].append(node_map[link['target']])
            link_data['value'].append(link['value'])
            
    node_colors = []
    for label in ordered_labels:
        if label in snp_names:
            node_colors.append('rgba(70, 186, 169, 0.8)')
        elif label in cpg_names:
            node_colors.append('rgba(238, 169, 104, 0.8)')
        elif label in gene_names:
            node_colors.append('rgba(148, 133, 192, 0.8)')
        else:
            node_colors.append('rgba(224, 96, 144, 0.8)')

    fig = go.Figure(data=[go.Sankey(
        node=dict(
          pad=15,
          thickness=20,
          line=dict(color="black", width=0.5),
          label=ordered_labels,
          color=node_colors
        ),
        link=dict(
          source=link_data['source'],
          target=link_data['target'],
          value=link_data['value']
      ))])

    fig.update_layout(title_text=f"Top {top_k} Connections per Layer in HINN", font_size=10)
    fig.show()


def load_and_prepare_data(target_column='MMSE'):
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

    data.dropna(subset=[target_column], inplace=True)
    
    demog_cols = ['AGE', 'GENDER', 'EDU', 'RACE', 'PTAU181', 'APOE']
    
    valid_targets = ['MMSE', 'MOCA', 'ADAS11', 'RAVLT.immediate']
    if target_column not in valid_targets:
        raise ValueError(f"Invalid target_column. Choose from: {valid_targets}")

    y = data[target_column].values.reshape(-1, 1)
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
    
    mask_gene_go = mask_gene_go_df.reindex(index=gene_features, columns=go_features, fill_value=0).values

    scaler_demog = StandardScaler()
    scaler_omics = StandardScaler()
    y_scaler = MinMaxScaler()

    X_demog_scaled = scaler_demog.fit_transform(X_demog)
    X_snp_scaled = scaler_omics.fit_transform(X_snp)
    X_cpg_scaled = scaler_omics.fit_transform(X_cpg)
    X_gene_scaled = scaler_omics.fit_transform(X_gene)
    y_scaled = y_scaler.fit_transform(y)

    X_snp_tensor = torch.FloatTensor(X_snp_scaled)
    X_cpg_tensor = torch.FloatTensor(X_cpg_scaled)
    X_gene_tensor = torch.FloatTensor(X_gene_scaled)
    X_demog_tensor = torch.FloatTensor(X_demog_scaled)
    y_tensor = torch.FloatTensor(y_scaled).squeeze()
    
    mask_snp_cpg_tensor = torch.from_numpy(mask_snp_cpg).float()
    mask_cpg_gene_tensor = torch.from_numpy(mask_cpg_gene).float()
    mask_gene_go_tensor = torch.from_numpy(mask_gene_go).float()
    
    return X_snp_tensor, X_cpg_tensor, X_gene_tensor, X_demog_tensor, y_tensor, \
           mask_snp_cpg_tensor, mask_cpg_gene_tensor, mask_gene_go_tensor, y_scaler, \
           snp_features, cpg_features, gene_features, go_features


if __name__ == '__main__':
    target_variable = 'MMSE' 
    
    result = load_and_prepare_data(target_column=target_variable)
    
    if result is None:
        print("Data loading failed.")
        exit()

    X_snp, X_cpg, X_gene, X_demog, y, \
    mask_snp_cpg, mask_cpg_gene, mask_gene_go, y_scaler, \
    snp_names, cpg_names, gene_names, go_names = result
    
    print(f"Number of samples: {len(y)}")
    
    indices = torch.arange(len(y))
    train_val_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)

    X_snp_train, X_cpg_train = X_snp[train_indices], X_cpg[train_indices]
    X_gene_train, X_demog_train = X_gene[train_indices], X_demog[train_indices]
    y_train = y[train_indices]
    
    X_snp_val, X_cpg_val = X_snp[val_indices], X_cpg[val_indices]
    X_gene_val, X_demog_val = X_gene[val_indices], X_demog[val_indices]
    y_val = y[val_indices]
    
    X_snp_test, X_cpg_test = X_snp[test_indices], X_cpg[test_indices]
    X_gene_test, X_demog_test = X_gene[test_indices], X_demog[test_indices]
    y_test = y[test_indices]
    

    train_dataset = TensorDataset(X_snp_train, X_cpg_train, X_gene_train, X_demog_train, y_train)
    val_dataset = TensorDataset(X_snp_val, X_cpg_val, X_gene_val, X_demog_val, y_val)
    test_dataset = TensorDataset(X_snp_test, X_cpg_test, X_gene_test, X_demog_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
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
        demog_dim=demog_dim,
        dropout_prob=0.7
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    num_epochs = 20
    best_val_loss = float('inf')
    
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
        
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for snp, cpg, gene, demog, labels in val_loader:
                outputs = model(snp, cpg, gene, demog)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    
    model.load_state_dict(best_model_state)
    

    model.eval()
    total_mae = 0
    total_mse = 0
    with torch.no_grad():
        for snp, cpg, gene, demog, labels in test_loader:
            outputs = model(snp, cpg, gene, demog)
            
            outputs_rescaled = y_scaler.inverse_transform(outputs.unsqueeze(1).numpy())
            labels_rescaled = y_scaler.inverse_transform(labels.unsqueeze(1).numpy())
            
            mse = ((outputs_rescaled - labels_rescaled) ** 2).mean()
            mae = np.abs(outputs_rescaled - labels_rescaled).mean()
            
            total_mse += mse * len(labels)
            total_mae += mae * len(labels)

    avg_mse = total_mse / len(test_dataset)
    avg_mae = total_mae / len(test_dataset)

    print(f'Mean Squared Error (MSE): {avg_mse:.4f}')
    print(f'Mean Absolute Error (MAE): {avg_mae:.4f}')

    plot_sankey(model, snp_names, cpg_names, gene_names, go_names, top_k=3)
