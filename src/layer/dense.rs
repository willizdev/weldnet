use crate::{math::Math, matrix::Matrix};

pub struct DenseLayer {
    weights: Matrix,
    biases: Matrix,
    inputs: Option<Matrix>,
    output: Option<Matrix>,
    // gradients
    d_weights: Option<Matrix>,
    d_biases: Option<Matrix>,
    d_inputs: Option<Matrix>,
}

impl DenseLayer {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        // weights are initialized with size (n_inputs, n_neurons)
        // as to skip the need for transposing during the forward pass
        let weights: Matrix = Matrix::random(n_inputs, n_neurons, -1.0, 1.0);
        let biases: Matrix = Matrix::zeros(1, n_neurons);

        Self {
            weights,
            biases,
            inputs: None,
            output: None,
            d_weights: None,
            d_biases: None,
            d_inputs: None,
        }
    }

    pub fn forward(&mut self, inputs: &Matrix) {
        self.output = Some(Math::add_row(
            &Math::product(inputs, &self.weights),
            &self.biases,
        ));
        // input is stored for backpropagation
        self.inputs = Some(inputs.clone());
    }

    pub fn backward(&mut self, d_values: &Matrix) {
        // some transposes are needed for correct dimensions
        self.d_weights = Some(Math::product(
            &Math::transpose(self.inputs.as_ref().unwrap()),
            d_values,
        ));
        self.d_biases = Some(Math::reduce_sum(d_values));
        self.d_inputs = Some(Math::product(d_values, &Math::transpose(&self.weights)));
    }
}
