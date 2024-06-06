import torch
import torch.nn as nn
from torch_geometric.utils import erdos_renyi_graph, to_dense_adj
from torch_geometric.data import Data, Batch
from torch_geometric.seed import seed_everything


def sample_batch(batch_size, num_nodes=8, edge_prob=0.3):
    data_list = [
        Data(
            edge_index=erdos_renyi_graph(num_nodes, edge_prob, directed=True),
            num_nodes=num_nodes,
        )
        for _ in range(batch_size)
    ]
    batch = Batch.from_data_list(data_list)
    return to_dense_adj(batch.edge_index, batch.batch)  # B, N, N


def compute_rwse(batch_adj: torch.Tensor, max_num_length: int):
    """
    Arguments:
        batch_adj: batched adjacency matrices of size B x N x N,
        where B is the batch size and N is the number of nodes
        max_num_length: Maximum random walk length

    Returns:
        Tensor of size B x N x max_num_length
        containing the random walk probabilities
    """
    # NOTE: 1e-5 for numerical stability
    inv_degrees = 1 / (batch_adj.sum(-1, keepdim=True) + 1e-5)  # B x N x 1
    # NOTE: left-multiplying a diagonal matrix is the same as multiplying
    # the i-th row with the i-th diagonal entry
    norm_adj = inv_degrees * batch_adj  # B x N x N

    probs = [torch.diagonal(norm_adj, dim1=-2, dim2=-1)]
    for _ in range(max_num_length):
        norm_adj = torch.bmm(norm_adj, norm_adj)
        probs.append(torch.diagonal(norm_adj, dim1=-2, dim2=-1))

    rwse_probs = torch.stack(probs, -1)  # B x N x max_num_length
    assert (
        not rwse_probs.isnan().any()
    ), "NaN probs, try to increase the edge probability of the random graphs"
    return rwse_probs


class MLP(nn.Sequential):
    def __init__(self, in_dim, embed_dim, out_dim):
        super().__init__(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )


class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout, num_probs, num_out):
        super().__init__()
        self.rwse_encoder = MLP(num_probs, embed_dim, embed_dim)
        self.transformer_encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    embed_dim, num_heads, embed_dim, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.head = MLP(embed_dim, embed_dim, num_out)

    def forward(self, rwse_probs: torch.Tensor):
        node_tokens = self.rwse_encoder(rwse_probs)

        for layer in self.transformer_encoder:
            node_tokens = layer(node_tokens)

        pooled_graph_token = node_tokens.sum(1)  # B x D
        return self.head(pooled_graph_token)


def main(max_num_length, num_nodes, verbose=False):
    """
    Arguments:
        max_num_length: The maximum random walk length to consider
        for this run
        num_nodes: Number of nodes in the random graphs
        verbose: If true, plot training stats at each step

    Returns:
        Test loss corresponding to best achieved training loss
    """
    num_steps = 200
    batch_size = 64
    model = Transformer(
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        dropout=0.0,
        num_probs=max_num_length + 1,
        # NOTE: We predict a single number for the mean degree task
        num_out=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def compute_loss():
        batch_adj = sample_batch(batch_size, num_nodes)
        rwse_probs = compute_rwse(batch_adj, max_num_length)

        # NOTE: Here, we regress over the mean degree of the graph
        targets = batch_adj.sum(1).mean(1, keepdim=True)  # B x 1

        preds = model(rwse_probs)

        # NOTE: We use MSE loss between predictions and targets
        return nn.functional.mse_loss(preds, targets)

    def train_step():
        loss = compute_loss()
        loss.backward()
        optimizer.step()
        return loss.item()

    best_train_loss = None
    for step in range(num_steps):
        model.train()
        optimizer.zero_grad()
        loss = train_step()

        if best_train_loss is None or loss < best_train_loss:
            best_train_loss = loss

            with torch.no_grad():
                model.eval()
                test_loss = compute_loss().item()

        if verbose:
            print(
                f"Step {step} | Train loss {float(loss):.2f} | Best train loss {float(best_train_loss):.2f} | Test loss {float(test_loss):.2f}"
            )

    return test_loss


if __name__ == "__main__":
    num_nodes = 8
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for max_num_length in [0, num_nodes // 2, num_nodes]:
        test_losses = []
        for seed in seeds:
            seed_everything(seed)
            test_losses += [main(max_num_length, num_nodes)]

        test_losses = torch.tensor(test_losses)
        avg_loss = test_losses.mean().item()
        std_loss = test_losses.std().item()
        print(
            f"RWSE length {max_num_length} | Avg. test loss {float(avg_loss):.3f} | Std. dev. test loss {float(std_loss):.3f}"
        )
