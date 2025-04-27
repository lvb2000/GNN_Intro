
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter
import torch_geometric.nn as pygnn
import torch_geometric.data as pygdata
from torch_geometric.utils import to_dense_batch, degree
from torch_geometric.graphgym import GNNGraphHead,BondEncoder
from mamba import Mamba, ModelArgs
import utils
from torch_geometric.graphgym.models.encoder import AtomEncoder
from laplace_pos_encoder import LapPENodeEncoder
from composed_encoders import concat_node_encoders

class GCN(torch.nn.Module):
    def __init__(self,node_features,classes):
        super().__init__()
        self.conv1 = pygnn.GCNConv(node_features, 16)
        self.conv2 = pygnn.GCNConv(16, classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        # Encode integer node features via nn.Embeddings
        NodeEncoder = concat_node_encoders([AtomEncoder, LapPENodeEncoder],['LapPE'])
        self.node_encoder = NodeEncoder(96)
        # Update dim_in to reflect the new dimension fo the node features
        self.dim_in = 96

        # Hard-set edge dim for PNA.
        self.edge_encoder = BondEncoder(96)


    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        # Loop over multiple GMB (Graph Mamba Layers)
        layers = []
        for _ in range(10):
            layers.append(GMBLayer(channels=96))
        self.layers = torch.nn.Sequential(*layers)

        self.post_mp = GNNGraphHead(dim_in=96, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch



"""
This function is extracted from GraphGPS Framework and customized to fit the Peptides-func framework.
These are the graph transformer settings for peptides:
gt:
  layer_type: CustomGatedGCN+Mamba_Hybrid_Degree_Noise (Local + Global Model Type)
  dim_hidden: 96  # `gt.dim_hidden` must match `gnn.dim_inner`
  dropout: 0.0
  attn_dropout: 0.5
  layer_norm: False
  batch_norm: True
"""
class GMBLayer(torch.nn.Module):

    def __init__(
        self,
        channels: int,
        equivstable_pe: bool = False,
        dropout: float = 0.0,
        order_by_degree: bool = False,
        shuffle_ind: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()

        self.channels = channels
        self.dropout = dropout
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        self.equivstable_pe = equivstable_pe
        
        assert (self.order_by_degree==True and self.shuffle_ind==0) or (self.order_by_degree==False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'
        
        # Local model type is the GatedGCNLayer
        self.local_model = GatedGCNLayer(channels, channels,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)

        # Global model t
        model_args = ModelArgs(d_model=channels,n_layer=4,d_state=d_state, d_conv=d_conv,expand=1)
        self.self_attn = Mamba(model_args)

        self.norm1_local = nn.BatchNorm1d(channels)
        self.norm1_attn = nn.BatchNorm1d(channels)

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        self.mlp =nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.BatchNorm1d(channels)


    def forward(self, batch):
        r"""Runs the forward pass of the module."""
        h = batch.x
        h_in1 = h  # for first residual connection
        h_out_list = []

        # Local Model forward pass
        local_out = self.local_model(pygdata.Batch( batch=batch,
                                                    x=h,
                                                    edge_index=batch.edge_index,
                                                    edge_attr=batch.edge_attr,
                                                    pe_EquivStableLapPE=self.equivstable_pe))
        # GatedGCN does residual connection and dropout internally.
        h_local = local_out.x
        batch.edge_attr = local_out.edge_attr

        # Apply batch norm
        h_local = self.norm1_local(h_local)

        # Append to forward path
        h_out_list.append(h_local)

        # Forward pass for global attention block (Mamba_Hybrid_Degree_Noise)
        h_dense, mask = to_dense_batch(h, batch.batch)
        if batch.split == 'train':
            # Get degree of each node in batch
            deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
            deg_noise = torch.rand_like(deg)
            # sort by degree
            h_ind_perm = utils.lexsort([deg+deg_noise, batch.batch])
            # apply sort
            h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
            # keep reverse sort to apply to output
            h_ind_perm_reverse = torch.argsort(h_ind_perm)
            # apply attention
            h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
        else:
            # for validation have 5 parallel attention layers
            mamba_arr = []
            deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
            for i in range(5):
                deg_noise = torch.rand_like(deg)
                h_ind_perm = utils.lexsort([deg+deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                mamba_arr.append(h_attn)
            h_attn = sum(mamba_arr) / 5

        # Add Dropout to global attention
        h_attn = self.dropout_attn(h_attn)
        # Residual connection
        h_attn = h_in1 + h_attn
        h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        # Combine local and global outputs.
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self.mlp(h)
        h = self.norm2(h)

        batch.x = h
        return batch

class GatedGCNLayer(pygnn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.A = pygnn.Linear(in_dim, out_dim, bias=True)
        self.B = pygnn.Linear(in_dim, out_dim, bias=True)
        self.C = pygnn.Linear(in_dim, out_dim, bias=True)
        self.D = pygnn.Linear(in_dim, out_dim, bias=True)
        self.E = pygnn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out