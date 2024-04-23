import torch
from monotonenorm import GroupSort, direct_norm, SigmaNet

class ExpressiveNetwork(torch.nn.Module):
    def __init__(self, width=128, sigma=1, monotone_constraints=[1], input_dim=1):
        super(ExpressiveNetwork, self).__init__()

        self.model = torch.nn.Sequential(
            direct_norm(torch.nn.Linear(input_dim, width), kind="one-inf"),
            GroupSort(width // 2),
            direct_norm(torch.nn.Linear(width, width), kind="inf"),
            GroupSort(width // 2),
            direct_norm(torch.nn.Linear(width, width), kind="inf"), #added 1601
            GroupSort(width // 2), #added 1601
            direct_norm(torch.nn.Linear(width, 1), kind="inf")
        )
        self.model = SigmaNet(self.model, sigma=sigma, monotone_constraints=monotone_constraints)

    def forward(self, x):
        return self.model(x)

    def count_params(self):
        return sum(p.numel() for p in self.model.nn.parameters() if p.requires_grad)


# def ExpressiveNetwork(width=128, sigma=1, monotone_constraints=[1], input_dim=1):
#     model = torch.nn.Sequential(
#         direct_norm(torch.nn.Linear(input_dim, width), kind="one-inf"),
#         GroupSort(width // 2),
#         direct_norm(torch.nn.Linear(width, width), kind="inf"),
#         GroupSort(width // 2),
#         direct_norm(torch.nn.Linear(width, width), kind="inf"), #added 1601
#         GroupSort(width // 2), #added 1601
#         direct_norm(torch.nn.Linear(width, 1), kind="inf"),
#     )
#     model = SigmaNet(model, sigma=sigma, monotone_constraints=monotone_constraints)
#     return model