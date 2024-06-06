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
    # TODO
    return None


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
        # TODO: Implement an encoder for your RWSE probabilities
        node_tokens = None

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
    # TODO (optional) adjust training hyper-parameters to suit your task
    # or based on a hyper-parameter search you conducted
    num_steps = 200
    batch_size = 64
    model = Transformer(

        # TODO (optional) adjust model hyper-parameters to suit your task
        # or based on a hyper-parameter search you conducted
        num_layers=2,
        embed_dim=32,
        num_heads=4,
        dropout=0.0,
        num_probs=max_num_length + 1,

        # TODO Set the output dimension as needed for your task
        num_out=None
    )

    # TODO (optional) adjust optimizer hyper-parameters to suit your task
    # or based on a hyper-parameter search you conducted
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def compute_loss():
        batch_adj = sample_batch(batch_size, num_nodes)
        rwse_probs = compute_rwse(batch_adj, max_num_length)

        # TODO: Define your task based on some graph property
        # (use `batch_adj`, which contains the graph structure
        # in the form of adjacency matrices, to compute your targets)
        targets = None
        preds = model(rwse_probs)

        # TODO: Compute your task loss between `targets` and `preds` 
        return None

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
    # TODO Select number of nodes you want for your task
    num_nodes = None
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
