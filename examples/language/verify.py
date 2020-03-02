import argparse, random, pickle, os, pdb, time
parser = argparse.ArgumentParser()

parser.add_argument('dir', type=str)
parser.add_argument('--max_budget', type=int, default=6)
parser.add_argument('--model', type=str, default="transformer", choices=["transformer"])

args = parser.parse_args()

def load_result(filename):
    with open(filename, "rb") as file:
        res = pickle.load(file)
    return res

def print_result(res, caption=None):
    if caption is not None:
        print(caption)
    print("acc {:.3f}, acc-robust {:.3f}".format(
        res[0].avg, res[2].avg))

def verify(method, budget):
    if method == "ibp":
        arguments = "--robust --ibp"
    elif method == "ibp+backward":
        arguments = "--robust --ibp --method=backward"
    elif method == "forward":
        arguments = "--robust --method=forward --batch_size=4"
    elif method == "forard+backward":
        arguments = "--robust --method=backward --batch_size=4"
    cmd = "python -m examples.language.train_language --dir={} {} --model={} --budget={} --res_file=out.txt".format(
        args.dir, arguments, args.model, budget)
    print(cmd)
    os.system(cmd)
    res = load_result("out.txt")
    print_result(res)
    return res

res = []
for budget in range(1, args.max_budget + 1):
    r = []
    r.append(verify("ibp", budget))
    r.append(verify("ibp+backward", budget))
    r.append(verify("forward", budget))
    r.append(verify("forward+backward", budget))
    res.append(r)

for budget in range(1, args.max_budget + 1):
    print_result(res[budget - 1][0], "budget = {}, ibp".format(budget))
    print_result(res[budget - 1][1], "budget = {}, ibp+backward".format(budget))
    print_result(res[budget - 1][2], "budget = {}, forward".format(budget))
    print_result(res[budget - 1][2], "budget = {}, forward+backward".format(budget))
