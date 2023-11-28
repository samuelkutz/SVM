# SVM (Support Vector Machine) in Julia

This Julia package provides a simple implementation of a Support Vector Machine (SVM) for classification tasks. The SVM is a supervised machine learning model that can be used for both linear and non-linear classification.

##### _We are working in turning this project into a package_

## Installation

You can simply clone the files using git clone:

```
git clone https://github.com/samuelkutz/svm-smo/
```


Once downloaded, you can simply include it in your julia code

```julia
include("path/to/clone/src/svm.jl")
```

Now you're ready to create and train Support Vector Machine models for your classification tasks!

## Usage

```julia
# Import the SVM module
include("your-path-to-clone/src/svm.jl")

# Create an SVM model with default parameters or your own
svm_model = SVM() 

# Load your training data (X) and labels (y)
X_train = ...
y_train = ...

# Fit the SVM model to the training data
fit!(svm_model, X_train, y_train)

# Make predictions on new data
x_new = ...
prediction = predict(svm_model, x_new)
```

## SVM Parameters

The SVM model is defined with the following parameters:

- `C`: Regularization parameter (default: 1.0)
- `tol`: Tolerance for stopping criterion (default: 1e-3)
- `kernel_type`: Type of kernel function to be used. Currently supports only the linear kernel.
- `gamma`: Kernel coefficient for 'rbf' and 'poly' kernels (default: 0.5)
- `degree`: Degree of the polynomial kernel function ('poly') (default: 3.0)
- `coef0`: Independent term in the kernel function ('poly' and 'sigmoid') (default: 0.0)

## SVM Methods

### `fit!(svm::SVM, X::Matrix{Float64}, y::Vector{Float64})`

Fit the SVM model to the provided training data.

### `predict(svm::SVM, x_new::Vector{Float64})`

Make predictions on new data using the trained SVM model.

## Kernel Functions

The package currently supports only the linear kernel. Additional kernel functions such as 'rbf', 'poly', and 'sigmoid' will be implemented in future releases.

## Contributions

Contributions and bug reports are welcome! Feel free to contact us in case of any doubts.
