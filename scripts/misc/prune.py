from torch import nn
from torch.nn.utils import prune

units = 3
layers = [nn.Linear(4, units), nn.ReLU()]

for i in range(1):
    layers.append(nn.Linear(in_features=units, out_features=units))
    layers.append((nn.ReLU()))
layers += [nn.Linear(units, 1), nn.ReLU()]
model = nn.Sequential(*layers)

if __name__ == '__main__':
    obj = dict(model.named_parameters())
    for param in obj.items():
        print(param)

    params = [
        (m, next(m.named_parameters())[0])
        for m in list(model.modules())[1:: 2]]
    prune.global_unstructured(params,
                              pruning_method=prune.L1Unstructured, amount=.3)

    for param in model.named_parameters():
        print(param)
