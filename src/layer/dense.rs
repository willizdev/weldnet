use crate::{math::Math, matrix::Matrix};

pub struct DenseLayer {
    weights: Matrix,
    biases: Matrix,
    output: Option<Matrix>,
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
            output: None,
        }
    }

    pub fn forward(&mut self, inputs: &Matrix) {
        self.output = Some(Math::add_row(
            &Math::product(inputs, &self.weights),
            &self.biases,
        ));
    }

    pub fn get_weights(&self) -> &Matrix {
        &self.weights
    }

    pub fn get_biases(&self) -> &Matrix {
        &self.biases
    }

    pub fn get_output(&self) -> &Option<Matrix> {
        &self.output
    }
}
