`auto_LiRPA` is a library for automatically deriving and computing bounds with
linear relaxation based perturbation analysis (LiRPA) (e.g.
[CROWN](https://arxiv.org/pdf/1811.00866.pdf) and
[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf)) for
neural networks, which is an useful tool for formal robustness verification. We
generalize existing LiRPA algorithms for feed-forward neural networks to a
graph algorithm on general computational graphs, defined by PyTorch.
Additionally, our implementation is also automatically **differentiable**,
allowing optimizing network parameters to shape the bounds into certain
specifications (e.g., certified defense). You can find [a video ▶️ introduction
here](http://PaperCode.cc/AutoLiRPA-Video).

Our library supports the following algorithms:

* Backward mode LiRPA bound propagation ([CROWN](https://arxiv.org/pdf/1811.00866.pdf)/[DeepPoly](https://files.sri.inf.ethz.ch/website/papers/DeepPoly.pdf))
* Backward mode LiRPA bound propagation with optimized bounds ([α-CROWN](https://arxiv.org/pdf/2011.13824.pdf))
* Backward mode LiRPA bound propagation with split constraints ([β-CROWN](https://arxiv.org/pdf/2103.06624.pdf))
* Forward mode LiRPA bound propagation ([Xu et al., 2020](https://arxiv.org/pdf/2002.12920))
* Forward mode LiRPA bound propagation with optimized bounds (similar to [α-CROWN](https://arxiv.org/pdf/2011.13824.pdf))
* Interval bound propagation ([IBP](https://arxiv.org/pdf/1810.12715.pdf))
* Hybrid approaches, e.g., Forward+Backward, IBP+Backward ([CROWN-IBP](https://arxiv.org/pdf/1906.06316.pdf))

Our library allows automatic bound derivation and computation for general
computational graphs, in a similar manner that gradients are obtained in modern
deep learning frameworks -- users only define the computation in a forward
pass, and `auto_LiRPA` traverses through the computational graph and derives
bounds for any nodes on the graph.  With `auto_LiRPA` we free users from
deriving and implementing LiPRA for most common tasks, and they can simply
apply LiPRA as a tool for their own applications.  This is especially useful
for users who are not experts of LiRPA and cannot derive these bounds manually
(LiRPA is significantly more complicated than backpropagation).