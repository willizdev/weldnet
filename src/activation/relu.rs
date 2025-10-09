use crate::matrix::Matrix;

pub struct ActFnRelu {
    inputs: Option<Matrix>,
    output: Option<Matrix>,
    // gradients
    d_inputs: Option<Matrix>,
}

impl ActFnRelu {
    pub fn new() -> Self {
        Self {
            inputs: None,
            output: None,
            d_inputs: None,
        }
    }

    pub fn forward(&mut self, inputs: &Matrix) {
        let mut output: Matrix = inputs.clone();

        for i in 0..output.rows {
            for j in 0..output.cols {
                if output.get(i, j) <= 0.0 {
                    output.set(i, j, 0.0);
                }
            }
        }

        self.output = Some(output);
        // input is stored for backpropagation
        self.inputs = Some(inputs.clone());
    }

    pub fn backward(&mut self, d_values: &Matrix) {
        let inputs: &Matrix = self.inputs.as_ref().unwrap();
        let mut d_inputs: Matrix = d_values.clone();

        for i in 0..d_inputs.rows {
            for j in 0..d_inputs.cols {
                if inputs.get(i, j) <= 0.0 {
                    d_inputs.set(i, j, 0.0);
                }
            }
        }

        self.d_inputs = Some(d_inputs);
    }

    pub fn get_output(&self) -> &Option<Matrix> {
        &self.output
    }
}
