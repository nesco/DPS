# DPS - Deep Program Search

## Structure

- localtypes.py: custom types used throughout the project
- grid.py: defines the main operations on grids
- helpers.py: helper functions
- freeman.py: defines the Freeman tree
- syntax_tree.py: defines the syntax trees, and operation on them
- lattice.py: defines the operation lattice

## Ideas:

> "What fire together wire together" (neurons) -> "What's close cluster together" (latent/semantics spaces)
=> Group things ""alike"" together

> Y. LeCun: Variance, Invariance, Covariance
=> Use Invariance to extract constance, and variance to extract functions
If there is structure (transitivity/clique), the function is parametrized by a Category
Else, back to standard information theory and find a distribution which encodes the distribution

> AIT vs IT
=> If there is structure, use a programming coding instead of a statistical one
