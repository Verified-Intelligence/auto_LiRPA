import random, sys, time, multiprocessing
from auto_LiRPA import BoundedModule, BoundedTensor, BoundedParameter
from auto_LiRPA.perturbations import *

class SimpleNet(nn.Module):
    def __init__(self, input_dim=11, output_dim=3, hidden_sizes=[64, 64],
                 activation=nn.Tanh):
        super().__init__()

        self.activation = activation()
        self.linear_layers = nn.ModuleList()
        self.weight_perturbations = []
        self.bias_perturbations = []
        prev_size = input_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i)
            # Replace weight and bias with their bounded versions
            ptb_w = PerturbationLpNorm(norm=np.inf, eps=0.)
            ptb_b = PerturbationLpNorm(norm=np.inf, eps=0.)
            lin.weight = BoundedParameter(lin.weight.data, ptb_w)
            lin.bias = BoundedParameter(lin.bias.data, ptb_b)
            # Save perturbation objects for further modification
            self.weight_perturbations.append(ptb_w)
            self.bias_perturbations.append(ptb_b)
            self.linear_layers.append(lin)
            prev_size = i

        self.final_linear = nn.Linear(prev_size, output_dim)
        ptb_w = PerturbationLpNorm(norm=np.inf, eps=0.)
        ptb_b = PerturbationLpNorm(norm=np.inf, eps=0.)
        self.final_linear.weight = BoundedParameter(self.final_linear.weight.data, ptb_w)
        self.final_linear.bias = BoundedParameter(self.final_linear.bias.data, ptb_w)
        self.weight_perturbations.append(ptb_w)
        self.bias_perturbations.append(ptb_b)
        

    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            # print('preactivation value layer', i)
            # print(x)
            x = self.activation(x)
        output = self.final_linear(x)
        return output.sum(dim=-1, keepdim=True)


def set_eps(model, p):
    for ptb in model.weight_perturbations:
        ptb.eps = p
    for ptb in model.bias_perturbations:
        ptb.eps = p


def compute_perturbations(model, x, perturbations):
    use_ibp = False
    method = 'backward'
    inputs = (x, )
    for p in perturbations:
        set_eps(model, p)
        lb, ub = model.compute_bounds(inputs, IBP=use_ibp, C=None, method=method, bound_lower=True, bound_upper=True)
        lb = lb.detach().cpu().numpy().squeeze()
        ub = ub.detach().cpu().numpy().squeeze()
        print("eps={:.4f}, lb={}, ub={}".format(p, lb, ub))
    set_eps(model, 0.0)
    lb, ub = model.compute_bounds(inputs, IBP=use_ibp, C=None, method=method, bound_lower=True, bound_upper=True)
    lb = lb.detach().cpu().numpy().squeeze()
    ub = ub.detach().cpu().numpy().squeeze()
    print("eps=0.0000\nlb={}\nub={}".format(lb, ub))


def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)
    input_size = 11

    ## Step 1: Initial original model as usual; note that this model has BoundedParameter as its weight parameters
    model_ori = SimpleNet(input_dim=input_size)

    ## Step 2: Prepare dataset as usual
    dummy_input1 = torch.randn(1, input_size)
    inputs = (dummy_input1, )

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter dummy_input is for constructing the trace of the computational graph.
    model = BoundedModule(model_ori, inputs)
    model.weight_perturbations = model_ori.weight_perturbations
    model.bias_perturbations = model_ori.bias_perturbations

    x = torch.randn(1, input_size)
    model(x)
    print('prediction', model_ori(x).squeeze().detach().cpu().numpy())
    compute_perturbations(model, x, np.linspace(0, 0.01, 5))
    # compute_perturbations(model, x, [])


if __name__ == "__main__":
    main()
