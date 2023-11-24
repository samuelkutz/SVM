# SVM-SMO

This project implements the Sequential Minimal Optimization (SMO) algorithm for training Support Vector Machines (SVM) in Julia.

## Installation

To install SVM-SMO, clone the repository and run the following command:

```
julia
```

Then, navigate to the project directory and run the following command:

```
include("svm-smo.jl")
```

## Usage

To train an SVM using the SMO algorithm, run the following command:

```
train_svm(data, C, kernel, degree, gamma)
```

where:

* `data` is a matrix of training data
* `C` is the regularization parameter
* `kernel` is the type of kernel function (e.g., "linear", "polynomial", or "rbf")
* `degree` is the degree of the polynomial kernel (only used if kernel is "polynomial")
* `gamma` is the gamma parameter of the RBF kernel (only used if kernel is "rbf")

## Output

The `train_svm` function will return a `svm_model` object, which contains the following information:

* `support_vectors`: The training examples that are closest to the decision boundary
* `dual_coefficients`: The coefficients of the support vectors in the dual representation of the SVM
* `bias`: The bias of the SVM

## License

This project is licensed under the MIT License.
