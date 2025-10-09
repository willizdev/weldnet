use crate::matrix::Matrix;

pub struct ActFnLinear {
    inputs: Option<Matrix>,
    output: Option<Matrix>,
    // gradients
    d_inputs: Option<Matrix>,
}

impl ActFnLinear {
    pub fn new() -> Self {
        Self {
            inputs: None,
            output: None,
            d_inputs: None,
        }
    }

    pub fn forward(&mut self, inputs: &Matrix) {
        let output: Matrix = inputs.clone();
        self.output = Some(output);
        // input is stored for backpropagation
        self.inputs = Some(inputs.clone());
    }

    pub fn backward(&mut self, d_values: &Matrix) {
        let d_inputs: Matrix = d_values.clone();
        self.d_inputs = Some(d_inputs);
    }

    pub fn get_output(&self) -> &Option<Matrix> {
        &self.output
    }
}
